# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;

# does mmd_AT generalize to mr 
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# double checkking on bert now that we changed so much, does mmd_at genralize to bert mr
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 1.0  --debug;

# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.0  --debug;





# CUDA_VISIBLE_DEVICES=2 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'AGNEWS_TextFooler_MMD' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 10 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'AGNEWS_TextFooler_MMD' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'AGNEWS_TextFooler_MMD' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

 
 

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --data_ratio 0.1  --debug;

# ONLINE

# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'Pooled_MR_Online' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 1 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'Pooled_MR_Online' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 1 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=6 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD_AT' --save_space 'Pooled_MR_Online' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 1 --batch_size 64 --data_ratio 0.1  --debug;

# baseline start
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023_Baseline' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 7 --batch_size 64 --data_ratio 0.0  --debug;

# baseline extended
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 7 --batch_size 64 --data_ratio 0.0  --debug;


#Base MMD 
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;

# mmd extended
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train BERTAttack --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train PWWS --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextBugger --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;

# AT
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;



# FLB
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.0  --debug;

# AUG
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=7 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
