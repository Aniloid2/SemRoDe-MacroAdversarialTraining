# CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Embedding_ablation' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;

CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'Dist' --method_type 'L2' --method_val 1 --save_space 'L2_ablation_mean' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'Dist' --method_type 'L2' --method_val 1 --save_space 'L2_ablation_mean' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'Dist' --method_type 'L2' --method_val 1 --save_space 'L2_ablation_mean' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'Dist' --method_type 'L2' --method_val 1 --save_space 'L2_ablation_mean' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;

