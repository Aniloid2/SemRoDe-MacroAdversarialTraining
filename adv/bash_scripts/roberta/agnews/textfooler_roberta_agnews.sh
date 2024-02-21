# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;


# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug; CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug --train False;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug --train False;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug; 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug; 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug --train False;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug; 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug --train False;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug; 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug ;


# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_AGNEWS_Online' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 1 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_AGNEWS_Online' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 1 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_AGNEWS_Online' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 1 --batch_size 64 --data_ratio 0.1  --debug;


# baseline start
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023_Baseline' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --batch_size 64 --data_ratio 0.0  --debug;

# baseline extended
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --batch_size 64 --data_ratio 0.0  --debug;


#Base MMD 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;

# mmd extended
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;

CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;

CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;

# AT
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;



# # FLB
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.0  --debug;

# AUG
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --online_epochs 3 --batch_size 64 --data_ratio 0.1  --debug;
