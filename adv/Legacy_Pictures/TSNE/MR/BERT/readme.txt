adv model


CUDA_VISIBLE_DEVICES=2 python analize_tsn.py --model  "/nas/Optimal_transport/GLUE/MR/BERT/Data_Hyper_Test/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val  0.01 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.2 --Epochs 10 --E 10 --NWS 0 --LR 0.00002 --Frozen "True" --Method 'epoch';


baseline
CUDA_VISIBLE_DEVICES=2 python analize_tsn.py --model  "/nas/Optimal_transport/GLUE/MR/BERT/50_and_10_epochs/at_stable/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val  0.0 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.0 --Epochs 10 --E 10 --NWS 0 --LR 0.000002 --Frozen "True" --Method 'epoch';
