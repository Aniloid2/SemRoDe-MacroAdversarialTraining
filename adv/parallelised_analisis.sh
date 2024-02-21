sleep 6; CUDA_VISIBLE_DEVICES=7 python analize_offline_test_scaled.py --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MMD_Test';
sleep 6; CUDA_VISIBLE_DEVICES=7 python analize_offline_test_scaled.py --dataset 'MRPC' --model 'BERT' --method 'CRL' --save_space 'CORAL_Test' ;
sleep 6; CUDA_VISIBLE_DEVICES=7 python analize_offline_test_scaled.py --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test' ;
