CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Hyper_Param_MMD_AT' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 1.0  --debug;