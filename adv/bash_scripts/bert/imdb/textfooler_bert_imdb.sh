# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'IMDB' --model 'BERT' --method 'FLB' --save_space 'Pooled_IMDB_Longer' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 10 --data_ratio 0  --debug;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'IMDB' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_IMDB_Longer_Smaller_BS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 10 --data_ratio 0.2  --debug;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'IMDB' --model 'BERT' --method 'AT' --save_space 'Pooled_IMDB_Longer_Smaller_BS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 10 --data_ratio 0.2  --debug ;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'IMDB' --model 'BERT' --method 'MMD' --save_space 'Pooled_IMDB_Longer_Smaller_BS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 10 --data_ratio 0.2  --debug;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'IMDB' --model 'BERT' --method 'B' --save_space 'Pooled_IMDB_Longer_Smaller_BS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 10 --data_ratio 0  --debug;

# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'IMDB' --model 'BERT' --method 'B' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'IMDB' --model 'BERT' --method 'FLB' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'IMDB' --model 'BERT' --method 'AT' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'IMDB' --model 'BERT' --method 'MMD' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'IMDB' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'IMDB' --model 'BERT' --method 'B' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'IMDB' --model 'BERT' --method 'FLB' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'IMDB' --model 'BERT' --method 'AT' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'IMDB' --model 'BERT' --method 'MMD' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'IMDB' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;




# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'IMDB' --model 'BERT' --method 'B' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'IMDB' --model 'BERT' --method 'FLB' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'IMDB' --model 'BERT' --method 'AT' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'IMDB' --model 'BERT' --method 'MMD' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'IMDB' --model 'BERT' --method 'MMD_AT' --save_space 'Pooled_IMDB' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;








# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
 