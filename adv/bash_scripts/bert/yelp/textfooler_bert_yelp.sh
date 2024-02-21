# CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'YELP' --model 'BERT' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'YELP' --model 'BERT' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'YELP' --model 'BERT' --method 'MMD_AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'YELP' --model 'BERT' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug --train 'False';
# CUDA_VISIBLE_DEVICES=1 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'YELP' --model 'BERT' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'YELP' --model 'BERT' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'YELP' --model 'BERT' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'YELP' --model 'BERT' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'YELP' --model 'BERT' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'YELP' --model 'BERT' --method 'MMD_AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;

# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'YELP' --model 'BERT' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'YELP' --model 'BERT' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'YELP' --model 'BERT' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'YELP' --model 'BERT' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'YELP' --model 'BERT' --method 'MMD_AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;


# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'YELP' --model 'BERT' --method 'B' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'YELP' --model 'BERT' --method 'FLB' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'YELP' --model 'BERT' --method 'AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'YELP' --model 'BERT' --method 'MMD' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=5 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'YELP' --model 'BERT' --method 'MMD_AT' --save_space 'ARR_OCT_2023' --GPU '1.1'  --frozen 'False' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug;








# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'True' --eval_method 'epoch' --epochs 3 --data_ratio 0  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=4 python ../../../high_level_at.py --attack_train TextFooler --attack_evaluate TextBugger --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'Pooled_AGNEWS' --GPU '1.1'  --frozen 'False' --eval_method 'epoch' --epochs 3 --data_ratio 0.1  --debug;
 