

# # BERT MR FREELB
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 10 --data_ratio 0 --eval_method 'epoch' --debug;

source_folder="./FREELB_LOCAL_Test/GLUE/MR/BERT/FreeLB_EMNLP2023/OT_GL_DMR_AKTTextFooler_ATETextFooler_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E10_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder1="./FREELB_LOCAL_Test/GLUE/MR/BERT/FreeLB_EMNLP2023/OT_GL_DMR_AKTTextFooler_ATEPWWS_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E10_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder2="./FREELB_LOCAL_Test/GLUE/MR/BERT/FreeLB_EMNLP2023/OT_GL_DMR_AKTTextFooler_ATEBERTAttack_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E10_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";

cp -r "$source_folder" "$destination_folder1";
cp -r "$source_folder" "$destination_folder2";

CUDA_VISIBLE_DEVICES=5 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 10 --data_ratio 0 --eval_method 'epoch' --debug --train False; 
CUDA_VISIBLE_DEVICES=4 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 10 --data_ratio 0 --eval_method 'epoch' --debug --train False;

# # Roberta MR FREELB
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 10 --data_ratio 0 --eval_method 'epoch' --debug;

source_folder="./FREELB_LOCAL_Test/GLUE/MR/ROBERTA/FreeLB_EMNLP2023/OT_GL_DMR_AKTTextFooler_ATETextFooler_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E10_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder1="./FREELB_LOCAL_Test/GLUE/MR/ROBERTA/FreeLB_EMNLP2023/OT_GL_DMR_AKTTextFooler_ATEPWWS_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E10_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder2="./FREELB_LOCAL_Test/GLUE/MR/ROBERTA/FreeLB_EMNLP2023/OT_GL_DMR_AKTTextFooler_ATEBERTAttack_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E10_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";

cp -r "$source_folder" "$destination_folder1";
cp -r "$source_folder" "$destination_folder2";

CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 10 --data_ratio 0 --eval_method 'epoch' --debug --train False;
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 10 --data_ratio 0 --eval_method 'epoch' --debug --train False;

# # # BERT AGNEWS FREELB
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False;

source_folder="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/FreeLB_EMNLP2023/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder1="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/FreeLB_EMNLP2023/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder2="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/FreeLB_EMNLP2023/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";

cp -r "$source_folder" "$destination_folder1";
cp -r "$source_folder" "$destination_folder2";


CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False; 
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False;

# # # ROBERTA AGNEWS FREELB
CUDA_VISIBLE_DEVICES=3 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False;

source_folder="./FREELB_LOCAL_Test/GLUE/AGNEWS/ROBERTA/FreeLB_EMNLP2023/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder1="./FREELB_LOCAL_Test/GLUE/AGNEWS/ROBERTA/FreeLB_EMNLP2023/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder2="./FREELB_LOCAL_Test/GLUE/AGNEWS/ROBERTA/FreeLB_EMNLP2023/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";

cp -r "$source_folder" "$destination_folder1";
cp -r "$source_folder" "$destination_folder2";


CUDA_VISIBLE_DEVICES=3 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False; CUDA_VISIBLE_DEVICES=3 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False; CUDA_VISIBLE_DEVICES=3 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'FLB' --save_space 'FreeLB_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0 --eval_method 'epoch' --debug --train False;

    

 