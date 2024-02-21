# sleep 6; CUDA_VISIBLE_DEVICES=0 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '0.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=0 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '0.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=0 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '0.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=0 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '0.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=1 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '1.1'  &
# sleep 6; CUDA_VISIBLE_DEVICES=1 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '1.2'  &
# sleep 6; CUDA_VISIBLE_DEVICES=1 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '1.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=1 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '1.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=2 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '2.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=2 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '2.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=2 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '2.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=2 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '2.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=3 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '3.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=3 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '3.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=3 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '3.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=3 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '3.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=4 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '4.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=4 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '4.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=4 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '4.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=4 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '4.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=5 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '5.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=5 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '5.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=5 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '5.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=5 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '5.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=6 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '6.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=6 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '6.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=6 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '6.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=6 python high_level_at.py --dataset 'MRPC' --model 'BERT' --GPU '6.4' &

# sleep 6; CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '7.1' &
# sleep 6; CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '7.2' &
# sleep 6; CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '7.3' &
# sleep 6; CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MR' --model 'BERT' --GPU '7.4' &

CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MR' --model 'BERT' --method 'OT' --save_space 'OT_Eval_Metrics_Test' --debug --GPU '7.1';


sleep 6; CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MR' --model 'BERT' --method 'CRL' --save_space 'CORAL_Test' --GPU '7.1';
sleep 6; CUDA_VISIBLE_DEVICES=7 python high_level_at.py --dataset 'MRPC' --model 'BERT' --method 'CRL' --save_space 'CORAL_Test' --GPU '6.1';
sleep 6; CUDA_VISIBLE_DEVICES=5 python high_level_at.py --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test' --GPU '5.1';
CUDA_VISIBLE_DEVICES=5 python high_level_at.py --dataset 'MRPC' --model 'BERT' --method 'AT' --save_space 'AT_Test' --GPU '5.1';

# sleep 6; CUDA_VISIBLE_DEVICES=0 python high_level_at.py --dataset 'MR' --model 'BERT' --method 'B' --save_space 'Baseline_Test' --GPU '0.0'; sleep 6; CUDA_VISIBLE_DEVICES=0 python high_level_at.py --dataset 'MRPC' --model 'BERT' --method 'B' --save_space 'Baseline_Test' --GPU '0.0';
