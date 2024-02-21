# MMD ROBERTA MR
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MR_TextBugger_MMD' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'MR_TextBugger_B' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 10 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'AGNEWS_TextBugger_B' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'AGNEWS_TextBugger_MMD' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug ; 
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug ;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;

# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'MR_TextBugger_B' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'B' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# baselines test
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
 
## train textbugger test on everython else mmd test 
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'TextBugger_Prototype' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# train on everything else, test on textbugger

CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


 