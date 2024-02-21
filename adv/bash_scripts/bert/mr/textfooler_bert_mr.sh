# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Broken_AT_With_Frozen' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Broken_AT_With_Frozen' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Broken_AT_With_Frozen' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'Broken_AT_With_Frozen' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'CRL' --save_space 'CRL_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'CRL' --save_space 'CRL_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'CRL' --save_space 'CRL_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'CRL' --save_space 'CRL_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'OT' --save_space 'OT_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'OT' --save_space 'OT_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'OT' --save_space 'OT_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'OT' --save_space 'OT_AAAI' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

#  ONLINE


CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;

CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;

CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;

CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextBugger --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'Redo_BERT_MR' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1 --debug;



# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'FreeLB_Distance_Calc' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;