#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Train a video classification model."""
import numpy as np
import pprint
from utils.utils import CheckpointSaver, get_current_time_str, load_checkpoint, log_test_result, remove_file, save_checkpoint
import torch
import torch.nn.functional as F
import math
import os
import oss2 as oss
import torch.nn as nn

import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu
from utils.meters import TrainMeter, ValMeter

from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

from datasets.utils.mixup import Mixup
from tqdm import tqdm
from dotenv import load_dotenv
import wandb
logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, writer=None, val_meter=None, val_loader=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    train_meter.iter_tic()
    
    data_size = cfg.SOLVER.STEPS_ITER    
    # STEP: TRAIN ITERATION
    ckpt_saver = CheckpointSaver(cfg.CHECKPOINT.NUM_TO_KEEP)
    last_ckpt_path = None
    loader_iter = iter(train_loader) 
    num_train_tasks = cfg.TRAIN.NUM_TRAIN_TASKS
    running_train_loss = []
    running_train_acc = []
    for cur_iter in tqdm(range(num_train_tasks)):
        wandb_log = {}
    # TRAIN ITERATION
        # STEP: MOVE INPUT TO GPU
        task_dict = next(loader_iter)
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0]
        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k].cuda(non_blocking=True)
        # MOVE INPUT TO GPU      
        
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', "real_support_labels"]'''
        # STEP: EVALUATE.
        val_log = None
        cur_epoch = cur_iter//cfg.SOLVER.STEPS_ITER
        # if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0 and cur_iter>=200:   # 
        if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0:   # 
            # if cfg.OSS.ENABLE:
            #     model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            #     model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            # else:
            #     model_bucket = None
            cur_epoch_save = cur_iter//cfg.TRAIN.VAL_FRE_ITER
            # cu.save_checkpoint(cfg.OUTPUT_DIR, model, model_ema, optimizer, cur_epoch_save+cfg.TRAIN.NUM_FOLDS-1, cfg, model_bucket)
            
            val_meter.set_model_ema_enabled(False)
            val_log = eval_epoch(val_loader, model, val_meter, cur_epoch_save+cfg.TRAIN.NUM_FOLDS-1, cfg, 'val', writer)
            
            wandb_log['val_loss'] = val_log['val_loss']
            wandb_log['val_acc'] = val_log['val_acc']
            
            # STEP: SAVE CHECKPOINT
            val_acc = val_log['val_acc']
            save_path = os.path.join(cfg.OUTPUT_DIR, f'it{cur_iter+1}_acc{val_acc:.2f}.pt')
            if ckpt_saver(val_log['val_acc'], save_path=save_path):
                save_checkpoint(save_path, model, optimizer, cur_iter+1)
            
            if last_ckpt_path != None:
                remove_file(last_ckpt_path)
            
            last_ckpt_path = os.path.join(cfg.OUTPUT_DIR, f'it{cur_iter+1}_acc{val_acc:.2f}_final.pt')
            save_checkpoint(last_ckpt_path, model, optimizer, cur_iter+1)
            
            model.train()

            # SAVE CHECKPOINT
            
            # if model_ema is not None:
            #     val_meter.set_model_ema_enabled(True)
            #     eval_epoch(val_loader, model_ema.module, val_meter, cur_epoch_save+cfg.TRAIN.NUM_FOLDS-1, cfg, writer)
            
            
        # EVALUATE

            

        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])


        # Update the learning rate.
        if not hasattr(cfg, 'checkpoint_loadpath'):
            # if not resume training
            lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)
        else:
            lr = optimizer.param_groups[-1]['lr']
        wandb_log["lr"] = lr
        
        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

        else:
            model_dict = model(task_dict)

        target_logits = model_dict['logits']

        if hasattr(cfg.TRAIN,"USE_CLASSIFICATION") and cfg.TRAIN.USE_CLASSIFICATION:
            if hasattr(cfg.TRAIN,"USE_CLASSIFICATION_ONLY") and cfg.TRAIN.USE_CLASSIFICATION_ONLY:
                loss = cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long()) /cfg.TRAIN.BATCH_SIZE
            elif hasattr(cfg.TRAIN,"USE_LOCAL") and cfg.TRAIN.USE_LOCAL:
                if hasattr(cfg.TRAIN,"TEMPORAL_LOSS_WEIGHT") and cfg.TRAIN.TEMPORAL_LOSS_WEIGHT:
                    # this is the loss
                    loss =  (cfg.TRAIN.TEMPORAL_LOSS_WEIGHT*model_dict["loss_temporal_regular"] + F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1).long())) /cfg.TRAIN.BATCH_SIZE
                else:
                    loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1).long())) /cfg.TRAIN.BATCH_SIZE
       
            else:
                loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE
        else:
            
            loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE
       
        # check Nan Loss.
        if math.isnan(loss):
            # logger.info(f"logits: {model_dict}")
            loss.backward(retain_graph=False)
            optimizer.zero_grad()
            continue
        loss.backward(retain_graph=False)
        
        if hasattr(cfg.TRAIN,"CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)

        # optimize
        if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
            optimizer.step()
            optimizer.zero_grad()
        # self.scheduler.step()

        if hasattr(cfg, "MULTI_MODAL") and\
            cfg.PRETRAIN.PROTOTYPE.ENABLE and\
            cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        # optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        if cfg.DETECTION.ENABLE or cfg.PRETRAIN.ENABLE:
            if misc.get_num_gpus(cfg) > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                None, None, loss, lr, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )
            if cfg.PRETRAIN.ENABLE:
                train_meter.update_custom_stats(loss_in_parts)

        else:
            # this is the branch
            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                if misc.get_num_gpus(cfg) > 1:
                    loss = du.all_reduce([loss])[0].item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = du.all_reduce([v])[0].item()
                else:
                    loss = loss.item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = v.item()
                train_meter.update_custom_stats(loss_in_parts)
                train_meter.update_custom_stats(top1_err_all)
                train_meter.update_custom_stats(top5_err_all)
            else:
                # this is the branch
                # Compute the errors.
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
                running_train_loss.append(loss)
                running_train_acc.append((100-top1_err)/100)
        

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                train_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        if (cur_iter + 1) % cfg.TRAIN.LOG_FREQ == 0:
            num_step = cfg.TRAIN.LOG_FREQ
            running_train_acc = torch.mean(torch.tensor(running_train_acc)).item()
            running_train_loss = torch.mean(torch.tensor(running_train_loss)).item()
            wandb_log[f"train_loss_{num_step}"] = running_train_loss
            wandb_log[f"train_acc_{num_step}"] = running_train_acc
            running_train_acc = []
            running_train_loss = []
        
        # LOG WANDB
        if not hasattr(cfg, 'debug') or not cfg.debug:
            wandb.log(wandb_log)
        
        # LOG WANDB
    
    if last_ckpt_path != None:
        remove_file(last_ckpt_path)
    last_ckpt_path = os.path.join(cfg.OUTPUT_DIR, f'it{cur_iter+1}_final.pt')
    save_checkpoint(last_ckpt_path, model, optimizer, cur_iter+1)

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch+cfg.TRAIN.NUM_FOLDS-1)
    train_meter.reset()
    
    return ckpt_saver


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, split, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    # STEP: VAL ITERATION
    if split == 'test':
        num_val_tasks = cfg.TRAIN.NUM_TEST_TASKS
    else:
        num_val_tasks = cfg.TRAIN.NUM_VAL_TASKS
    
    loader_iter = iter(val_loader) 
    
    val_loss = []
    val_acc = []
    for cur_iter in tqdm(range(num_val_tasks)):
        task_dict = next(loader_iter)
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0]
        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k].cuda(non_blocking=True)
    # VAL ITERATION

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if misc.get_num_gpus(cfg):
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if misc.get_num_gpus(cfg) > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        elif cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'PCMGenerator'):
            preds, logits = model(inputs)
            if "move_x" in preds.keys():
                preds["move_joint"] = preds["move_x"]
            elif "move_y" in preds.keys():
                preds["move_joint"] = preds["move_y"]
            num_topks_correct = metrics.topks_correct(preds["move_joint"], labels["self-supervised"]["move_joint"].reshape(preds["move_joint"].shape[0]), (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds["move_joint"].shape[0]) * 100.0 for x in num_topks_correct
            ]
            if misc.get_num_gpus(cfg) > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err,
                top5_err,
                preds["move_joint"].shape[0]
                * max(
                    misc.get_num_gpus(cfg), 1
                ),
            )
            val_meter.update_predictions(preds, labels)
        else:
            # this is the branch
            # preds, logits = model(inputs)
            model_dict = model(task_dict)

            # loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)
            target_logits = model_dict['logits']
            loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE


            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # this is the branch
                # Compute the errors.
                labels = task_dict['target_labels']
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
                # STEP UPDATE LOSS AND ACC
                val_loss.append(loss)
                val_acc.append((100-top1_err)/100)
                # UPDATE LOSS AND ACC
            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                val_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if misc.get_num_gpus(cfg):
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()
    # STEP: COMPUTE AVARAGE VAL LOSS AND ACC
    val_loss = torch.mean(torch.tensor(val_loss)).item()
    val_acc = torch.mean(torch.tensor(val_acc)).item()
    return {"val_loss": val_loss, "val_acc":val_acc}
    # COMPUTE AVARAGE VAL LOSS AND ACC

def init_session(cfg):
    # method
    run_name = 'hyrsm'
    if hasattr(cfg, "distance_type"):
        run_name += f'_{cfg.distance_type}'    
    # num shot
    num_shot = cfg.TRAIN.SHOT
    
    run_name = f'{run_name}/{num_shot}shot'
    
    if (hasattr(cfg, 'debug') and cfg.debug) or (hasattr(cfg, 'test_only') and cfg.test_only):
        cfg.OUTPUT_DIR = 'debug'
    else:
        wandb.login(key='835e03fed77f3418a3bcf4cd09a93bf951d32b91')
        wandb.init(
            entity="aiotlab", 
            project="few-shot-action-recognition", 
            group='sdtw vs tpm', name=run_name
        )
        
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, run_name, get_current_time_str())
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.TRAIN.LOG_FILE = 'log.txt'


def train_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    cfg.TRAIN.NUM_CLASS = len(cfg.TRAIN.CLASS_NAME)
    if hasattr(cfg, 'distance_type') and cfg.distance_type == 'tpm':
        cfg.tempo_prior = float(cfg.tempo_prior)
    
    init_session(cfg)
    # Set up environment.
    # du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    # Setup logging format.
    # TODO: update cfg.TRAIN.LOG_FILE
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    # TODO: uncomment the following line
    model, model_ema = build_model(cfg)

    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Load a checkpoint to resume training if applicable.
    # start_epoch = cu.load_train_checkpoint(cfg, model, model_ema, optimizer, model_bucket)
    if hasattr(cfg, 'checkpoint_loadpath'):
        load_checkpoint(cfg.checkpoint_loadpath, model, optimizer)
    start_epoch = 0
    
    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val") if cfg.TRAIN.EVAL_PERIOD != 0 else None  # val
    test_loader = build_loader(cfg, 'test')
    
    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        pass
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.SOLVER.MAX_EPOCH-start_epoch)%cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    cur_epoch = 0
    shuffle_dataset(train_loader, cur_epoch)
    if not hasattr(cfg, 'test_only') or not cfg.test_only:
        ckpt_saver = train_epoch(
            train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, writer, val_meter, val_loader
        )
        best_ckpt_path = ckpt_saver.get_best_checkpoint_path()
        load_checkpoint(best_ckpt_path, model)
    else:
        best_ckpt_path = cfg.checkpoint_loadpath
        load_checkpoint(cfg.checkpoint_loadpath, model)  
    
    res = eval_epoch(
        test_loader, model, val_meter, cur_epoch, cfg, 'test', writer
    )
    
    log_test_result(
        cfg, best_ckpt_path, res['val_acc']
    ) 
    
    # torch.cuda.empty_cache()
    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

