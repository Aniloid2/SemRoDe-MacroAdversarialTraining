

# ROBERTA AGNEWS AUG adv
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# source_folder="/nas/Optimal_transport/GLUE/AGNEWS/ROBERTA/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder1="/nas/Optimal_transport/GLUE/AGNEWS/ROBERTA/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder2="/nas/Optimal_transport/GLUE/AGNEWS/ROBERTA/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"

# cp -r "$source_folder" "$destination_folder1"
# cp -r "$source_folder" "$destination_folder2"

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # ROBERTA AGNEWS AT adv
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# source_folder="/nas/Optimal_transport/GLUE/AGNEWS/ROBERTA/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder1="/nas/Optimal_transport/GLUE/AGNEWS/ROBERTA/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder2="/nas/Optimal_transport/GLUE/AGNEWS/ROBERTA/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"

# cp -r "$source_folder" "$destination_folder1"
# cp -r "$source_folder" "$destination_folder2"

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT AGNEWS AUG adv
CUDA_VISIBLE_DEVICES=7 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug;

# source_folder="/nas/Optimal_transport/GLUE/AGNEWS/BERT/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder1="/nas/Optimal_transport/GLUE/AGNEWS/BERT/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder2="/nas/Optimal_transport/GLUE/AGNEWS/BERT/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"

# cp -r "$source_folder" "$destination_folder1"
# cp -r "$source_folder" "$destination_folder2"

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT AGNEWS AT adv
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug;

# source_folder="/nas/Optimal_transport/GLUE/AGNEWS/BERT/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder1="/nas/Optimal_transport/GLUE/AGNEWS/BERT/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"
# destination_folder2="/nas/Optimal_transport/GLUE/AGNEWS/BERT/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_DR0_1_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/"

# cp -r "$source_folder" "$destination_folder1"
# cp -r "$source_folder" "$destination_folder2"

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug --train False;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;




# # ROBERTA MR AUG adv
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
CUDA_VISIBLE_DEVICES=5 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # ROBERTA MR AT adv
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
CUDA_VISIBLE_DEVICES=5 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT MR AUG adv
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT MR AT adv
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;














# # BERT AGNEWS AT adv
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug;

source_folder="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF";
destination_folder1="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder2="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/AT_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B1_0_AT1_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";

cp -r "$source_folder" "$destination_folder1"
cp -r "$source_folder" "$destination_folder2"

CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug --train False; CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug --train False;

## AUG
CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug;

source_folder="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF";
destination_folder1="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEPWWS_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";
destination_folder2="./FREELB_LOCAL_Test/GLUE/AGNEWS/BERT/AUG_Test_EMNLP2023_DR0_1/OT_GL_DAGNEWS_AKTTextFooler_ATEBERTAttack_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGAdv_P0_0_PF0_0_FLB0_0_DR0_1_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/";

cp -r "$source_folder" "$destination_folder1"
cp -r "$source_folder" "$destination_folder2"

CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug --train False; CUDA_VISIBLE_DEVICES=6 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 3 --data_ratio 0.1 --eval_method 'epoch' --debug --train False;




CUDA_VISIBLE_DEVICES=5 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug; CUDA_VISIBLE_DEVICES=5 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;


CUDA_VISIBLE_DEVICES=4 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug; CUDA_VISIBLE_DEVICES=4 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023_DR0_1' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
