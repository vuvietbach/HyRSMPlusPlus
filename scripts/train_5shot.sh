python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type sdtw && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type tpm tempo_prior 2.0

python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml checkpoint_loadpath output/hyrsm/5shot/17-02_23-24/it10000_final.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type sdtw checkpoint_loadpath output/hyrsm/5shot/18-02_04-18/it10000_final.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type tpm tempo_prior 2.0 checkpoint_loadpath output/hyrsm_tpm/5shot/18-02_09-24/it10000_acc0.75.pt

python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type tpm tempo_prior 2.0 checkpoint_loadpath output/hyrsm_tpm/5shot/19-02_09-30/it2200_acc0.76.pt test_only True
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type tpm tempo_prior 2.0 test_only True  checkpoint_loadpath output/hyrsm_tpm/5shot/19-02_09-30/it1600_acc0.76.pt test_only True 

python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type tpm tempo_prior 2.0 test_only True  checkpoint_loadpath output/hyrsm_tpm/5shot/18-02_09-24/it10000_acc0.75.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type sdtw test_only True checkpoint_loadpath output/hyrsm_sdtw/5shot/19-02_04-27/it2000_acc0.76.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml test_only True checkpoint_loadpath output/hyrsm_original/5shot/18-02_23-40/it2200_acc0.76.pt

python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml test_only True checkpoint_loadpath output/hyrsm_original/5shot/18-02_23-40/it4800_acc0.75.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml test_only True checkpoint_loadpath output/hyrsm_original/5shot/18-02_23-40/it2000_acc0.76.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type sdtw test_only True checkpoint_loadpath output/hyrsm_sdtw/5shot/19-02_04-27/it2200_acc0.75.pt && \
python run.py --cfg configs/projects/hyrsmplusplus/ssv2_full/HyRSMplusplus_SSv2_Full_5shot_v1.yaml distance_type sdtw test_only True checkpoint_loadpath output/hyrsm_sdtw/5shot/19-02_04-27/it4800_acc0.75.pt

