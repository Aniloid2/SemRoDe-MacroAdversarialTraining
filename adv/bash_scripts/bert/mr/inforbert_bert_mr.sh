# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B'  --save_space 'NAACL_Baseline' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;



# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;

# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert_v2' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert_v2' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert_v2' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'Embedding' --method_type 'InfoBert' --method_val 1 --save_space 'InfoBert_v2' --GPU '1.1' --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;