CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'B' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;

CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'B' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;

CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'B' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'Pooled_MR_New_Baselines' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;




# # baselines test
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;


# ## train textbugger test on everython else mmd test 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# # train on everything else, test on textbugger

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


 