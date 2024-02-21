
# CUDA_VISIBLE_DEVICES=0 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_adv_baseline"


# CUDA_VISIBLE_DEVICES=0 python textattack_run_glue.py --baseline 'False' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT"
# CUDA_VISIBLE_DEVICES=0 python textattack_run_glue.py --baseline 'True' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "Original_baseline"



# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/Original_adv_learn/last_model/" --id "Original_adv_learn" --output_name "Original_adv_learn"
# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "textattack/bert-base-uncased-MRPC" --id "Original" --output_name "Original"
# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_baseline/last_model/" --id "OT_baseline" --output_name "OT_adv_baseline"


#### Original_Baseline ####
# we don't need to train since it's already trained as we copied this from alignment_attention_temp
# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/Original_baseline/last_model/" --id "Original_baseline" --output_name "Original_baseline"

#### Original adv learning ###
# This is with adversarial data augmentation
# we don't need to train since it's already trained as we copied this from alignment_attention_temp
# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/Original_adv_learn/last_model/" --id "Original_adv_learn" --output_name "Original_adv_learn"

#### OT_baseline ####
# this is by useing adv_dataset with the original training_dataset posing as the adv dataset and no adv_loss
# CUDA_VISIBLE_DEVICES=0 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_baseline"
# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_baseline/last_model/" --id "OT_baseline" --output_name "OT_baseline"

#### OT_adv_baseline ####
# this is by useing adv_dataset with the original training_dataset posing as the adv dataset together with adv_loss
# CUDA_VISIBLE_DEVICES=0 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_adv_baseline"
# CUDA_VISIBLE_DEVICES=6 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_adv_baseline/last_model/" --id "OT_adv_baseline" --output_name "OT_adv_baseline"


#### OT_adv_no_ot ####
# this is by useing the actual adv_dataset with 0.05 examples that have been attacked, these examples are used to generate regularizer with adv_loss
# CUDA_VISIBLE_DEVICES=6 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_adv_no_ot"
# CUDA_VISIBLE_DEVICES=6 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_adv_no_ot/last_model/" --id "OT_adv_no_ot" --output_name "OT_adv_no_ot"


#### OT_no_adv_yes_ot ####
#  this is by useing the actual adv_dataset with 0.05 examples that have been attacked, these examples are used to generate the s_loss and da_loss, but we don't se it for adv_loss
# this is basicly jumbot with adversarial training, this should work
# CUDA_VISIBLE_DEVICES=6 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_no_adv_yes_ot"
# CUDA_VISIBLE_DEVICES=6 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_no_adv_yes_ot/last_model/" --id "OT_no_adv_yes_ot" --output_name "OT_no_adv_yes_ot"


#### OT_yes_adv_yes_ot ####
# this is by useing the actual adv_dataset with 0.05 examples that have been attacked, these examples are used to generate the s_loss and da_loss, but we don't se it for adv_loss
# this is basicly jumbot with adversarial training, this should work
# CUDA_VISIBLE_DEVICES=6 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_yes_adv_yes_ot_athird_da_loss"
# CUDA_VISIBLE_DEVICES=6 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_yes_adv_yes_ot_athird_da_loss/last_model/" --id "OT_yes_adv_yes_ot_athird_da_loss" --output_name "OT_yes_adv_yes_ot_athird_da_loss"


#### OT_yes_adv_yes_ot ####
# do training with only 1 sample at start of training
# CUDA_VISIBLE_DEVICES=6 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_yes_adv_no_ot_2s"
# CUDA_VISIBLE_DEVICES=6 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_yes_adv_no_ot_2s/last_model/" --id "OT_yes_adv_no_ot_2s" --output_name "OT_yes_adv_no_ot_2s"



#### OT_yes_adv_no_ot ####
# do training with only 1 sample from each class at start of training
# CUDA_VISIBLE_DEVICES=7 python textattack_run_glue.py --baseline 'optimal_transport' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_yes_adv_no_ot_2s"
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/OT_yes_adv_no_ot_2s/last_model/" --id "OT_yes_adv_no_ot_2s" --output_name "OT_yes_adv_no_ot_2s"




#### OT_yes_adv_no_ot ####
# do training with only 1 sample from each class at start of training
# CUDA_VISIBLE_DEVICES=0 python textattack_run_glue.py --baseline 'optimal_transport_geomloss' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "opt_tran_geo_base"
# CUDA_VISIBLE_DEVICES=0 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/opt_tran_geo_base/last_model/" --id "opt_tran_geo_base" --output_name "opt_tran_geo_base"



#### OT dataset test ####
#pure baseline
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 0 --OT_Val 0 --Data_ratio 0;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0 --Data_ratio 0;

#baseline with at dataratio 0.2
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.2;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.2;

#OT with dataratio 0.2
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.2;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.2;

#OT and AT with dataratio 0.2
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.2;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.2;


#baseline with at dataratio 0.05
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.05;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.05;
#
# #OT with dataratio 0.05
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05;
#
# #OT and AT with dataratio 0.05
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05;
#
#
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.2;
#
#
#
# #
# # #baseline with at dataratio 0.5
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.5;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.5;
# #
# # #OT with dataratio 0.5
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.5;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.5;
# #
# # #OT and AT with dataratio 0.5
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.5;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.5;
# #
# # #baseline with at dataratio 1
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 1.0;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 1.0;
# #
# # #OT with dataratio 1
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 1.0;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 1.0;
# #
# # #OT and AT with dataratio 1
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 1.0;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0.01 --Data_ratio 1.0;



# OT over epochs
# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 2;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 2;

# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.05 --Epochs 2;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 1 --OT_Val 0 --Data_ratio 0.05 --Epochs 2;

# CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL" --output_name "OT_GL" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;


# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;

# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 2;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 2;

# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --AT_Val 0 --OT_Val 0.001 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --AT_Val 0 --OT_Val 0.001 --Data_ratio 0.05 --Epochs 1;

# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --AT_Val 0 --OT_Val 0.001 --Data_ratio 0.05 --Epochs 2;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --AT_Val 0 --OT_Val 0.001 --Data_ratio 0.05 --Epochs 2;


# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;


# OT no CE for original class
# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --B_Val 0 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --B_Val 0 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;

# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --id "OT_GL_CC" --B_Val 0 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --output_name "OT_GL_CC" --B_Val 0 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;


#mnli test
# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/bert-base-uncased-MNLI" --Dataset 'MNLI' --id "OT_GL_CC" --B_Val 0 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MRPC/ALBERT/" --id "OT_GL_CC" --Dataset 'MNLI' --output_name "OT_GL_CC" --B_Val 0 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;


# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "prajjwal1/albert-base-v2-mnli" --Dataset 'MNLI' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MNLI/ALBERT/" --id "OT_GL_CC" --Dataset 'MNLI' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 1;

# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "prajjwal1/albert-base-v2-mnli" --Dataset 'MNLI' --id "OT_GL_CC" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.1 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MNLI/ALBERT/" --id "OT_GL_CC" --Dataset 'MNLI' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.1 --Epochs 1;
#
#
# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "prajjwal1/albert-base-v2-mnli" --Dataset 'MNLI' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/MNLI/ALBERT/" --id "OT_GL_CC" --Dataset 'MNLI' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;


# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "prajjwal1/albert-base-v2-mnli" --id "Raw" --Dataset 'MNLI' --output_name "Raw" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;

# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/albert-base-v2-snli" --Dataset 'SNLI' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/SNLI/ALBERT/" --id "OT_GL_CC" --Dataset 'SNLI' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 1;
#
# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/albert-base-v2-snli" --Dataset 'SNLI' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/SNLI/ALBERT/" --id "OT_GL_CC" --Dataset 'SNLI' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 1;


# CUDA_VISIBLE_DEVICES=3  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/albert-base-v2-snli" --Dataset 'SNLI' --id "OT_GL_CC" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=3 python attack_glue.py --model  "./GLUE/SNLI/ALBERT/" --id "OT_GL_CC" --Dataset 'SNLI' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;

# ot cc mr
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/albert-base-v2-rotten-tomatoes" --Dataset 'MR' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/ALBERT/" --id "OT_GL_CC" --Dataset 'MR' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;

# raw attack on main model
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "textattack/albert-base-v2-rotten-tomatoes" --id "Raw" --Dataset 'MR' --output_name "Raw" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;

# adv baseline mr
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/albert-base-v2-rotten-tomatoes" --Dataset 'MR' --id "OT_GL_CC" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/ALBERT/" --id "OT_GL_CC" --Dataset 'MR' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;

# baseline mr
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/albert-base-v2-rotten-tomatoes" --Dataset 'MR' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/ALBERT/" --id "OT_GL_CC" --Dataset 'MR' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;


# # ot mr
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/albert-base-v2-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/ALBERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
#
# # ot mr with at
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/albert-base-v2-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/ALBERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;


# try with bert and freezing bert

# raw test on bert

# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "textattack/bert-base-uncased-rotten-tomatoes" --id "Raw" --Dataset 'MR' --output_name "Raw" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;



# ot mr BERT
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;

# baseline mr

# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 1;

# adv baseline mr BERT
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;

# ot mr BERT
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
#
# baseline mr
#
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 4;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0 --Epochs 4;
#
# adv baseline mr BERT \\\\\\\\\
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 4 --BS 64 --AS 1 --NWS 100 --LR 0.01 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4;
#
# ot cc mr BERT
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL_CC" --Dataset 'MR' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 1;
#
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL_CC' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL_CC" --Dataset 'MR' --output_name "OT_GL_CC" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
# ot mr BERT
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --id "OT_GL"  --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR'  --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4 --AEI 4 --BS 64 --AS 1 --NWS 100 --LR 0.01 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
#
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;


# MRPC v2 Baseline epoch wise online training

#Raw MRPC

# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "textattack/bert-base-uncased-MRPC" --id "Raw" --Dataset 'MRPC' --output_name "Raw" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 1;
#
# # baseline MRPC epoch wise online 4 epochs
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-MRPC" --Dataset 'MRPC' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.01 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/BERT/" --id "OT_GL" --Dataset 'MRPC' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 4;
#
# # baseline ot mrpc epoch wise online 4 epochs
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-MRPC" --Dataset 'MRPC' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.01 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/BERT/" --id "OT_GL" --Dataset 'MRPC' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4;
#
# # baseline at mrpc epoch wise online 4 epochs
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-MRPC" --Dataset 'MRPC' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.01 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/BERT/" --id "OT_GL" --Dataset 'MRPC' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4;


# baseline at MR oneline 4 epochs with smaller learning rate
# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 2e-5 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4;

# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "../original_results/GLUE/MRPC/ALBERT/Original_40_run_prime" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 2e-5 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4;


# CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.01 --WD 0.01;


# online training on best LR and NWS MR, MRPC
# MR
# # baseline
# # CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.001 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 4 --NWS 100 --LR 0.001;

# #AT baseline
# # CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.001 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --NWS 100 --LR 0.001;

# # OT
# # CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.001 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4 --NWS 100 --LR 0.001;

# # MRPC

# # baseline
# # CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-MRPC" --Dataset 'MRPC' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.05 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/BERT/" --id "OT_GL" --Dataset 'MRPC' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.0 --Data_ratio 0.0 --Epochs 4  --NWS 100 --LR 0.05;

# #AT baseline
# # CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-MRPC" --Dataset 'MRPC' --id "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.05 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/BERT/" --id "OT_GL" --Dataset 'MRPC' --output_name "OT_GL" --B_Val 1 --AT_Val 1 --OT_Val 0.0 --Data_ratio 0.05 --Epochs 4  --NWS 100 --LR 0.05;

# # OT
# # CUDA_VISIBLE_DEVICES=7  python textattack_run_glue.py --baseline 'OT_GL' --model  "textattack/bert-base-uncased-MRPC" --Dataset 'MRPC' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4 --AEI 1 --BS 64 --AS 1 --NWS 100 --LR 0.05 --WD 0.01;
# CUDA_VISIBLE_DEVICES=7 python attack_glue.py --model  "./GLUE/MRPC/BERT/" --id "OT_GL" --Dataset 'MRPC' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --Data_ratio 0.05 --Epochs 4  --NWS 100 --LR 0.05;



# # high level function MR ROBERTA GENERALIZATION
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True'  --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug    ; 
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True'  --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug   ;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True'  --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug    ;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'   --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'   --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;

# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'MMD_Test' --GPU '1.1'   --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'B' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;


# # MMD ROBERTA AGNEWS
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug ; 
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug ;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;

# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'B' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;

# # high level function MR BERT GENERALIZATION
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug  ; 
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug ;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug  ;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'B' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'B' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# # CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'B' --save_space 'MR_0_1_Test_PWWS_New_Double_Check_ModelEval' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;


# MMD BERT AGNEWS
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1  --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1  --debug;
CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1'  --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug ; 
CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug ;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'MMD' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;
CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 3 --data_ratio 0.1 --debug --train False;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'B' --save_space 'AGNEWS_Test' --GPU '1.1' --frozen 'True' --eval_method 'last' --epochs 10 --data_ratio 0.1 --debug;




# # ROBERTA MR AUG adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # ROBERTA MR AT adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT MR AUG adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT MR AT adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'MR' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;


# ROBERTA AGNEWS AUG adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # ROBERTA AGNEWS AT adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'ROBERTA' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT AGNEWS AUG adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AUG' --save_space 'AUG_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# # BERT AGNEWS AT adv
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train TextFooler --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train PWWS --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;

# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate TextFooler --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate PWWS --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;
# CUDA_VISIBLE_DEVICES=2 python high_level_at.py --attack_train BERTAttack --attack_evaluate BERTAttack --dataset 'AGNEWS' --model 'BERT' --method 'AT' --save_space 'AT_Test_EMNLP2023' --GPU '1.1' --frozen 'False' --epochs 7 --data_ratio 0.1 --eval_method 'last' --debug;