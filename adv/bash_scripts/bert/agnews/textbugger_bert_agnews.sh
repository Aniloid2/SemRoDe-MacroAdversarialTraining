CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'FLB' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
 



#  ## train textbugger test on everython else mmd test 
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;


# # train on everything else, test on textbugger

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
