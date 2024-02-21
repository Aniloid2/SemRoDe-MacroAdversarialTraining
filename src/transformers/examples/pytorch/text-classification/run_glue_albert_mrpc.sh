
# for (( c=0; c<=40; c++ ))
# do
#
# GLUE_DIR=../../../../../../huggingface/GLUE
# TASK_NAME=MRPC
# ID=Original_40_run_prime
# RES_LOC=../../../../../results/GLUE/$TASK_NAME/ALBERT
#
# python run_glue.py \
#   --local_rank -1 \
#   --seed $c \
#   --model_name_or_path albert-base-v2 \
#   --task_name  $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --gradient_accumulation_steps 1\
#   --learning_rate 2e-5 \
#   --num_train_epochs 4.0 \
#   --overwrite_output_dir \
#   --output_dir ../../../../../original_results/GLUE/$TASK_NAME/ALBERT/$ID/ \
#   --att_se_hid_size 16\
#   --att_se_nonlinear relu\
#   --rho 0.5 \
#   --adver_type nan \
#   --att_prior_type 'constant' \
#   --alpha_gamma 1.0 \
#   --att_kl 0.01 \
#
# GLUE_DIR=../../../../../../huggingface/GLUE
# TASK_NAME=MRPC
# ID=ACT_40_run_prime
# RES_LOC=../../../../../results/GLUE/$TASK_NAME/ALBERT
#
#
# python run_glue.py \
#     --local_rank -1 \
#     --seed $c \
#     --model_name_or_path albert-base-v2 \
#     --task_name  $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1\
#     --learning_rate 2e-5 \
#     --num_train_epochs 4.0 \
#     --overwrite_output_dir \
#     --output_dir ../../../../../original_results/GLUE/$TASK_NAME/ALBERT/$ID/ \
#     --att_se_hid_size 16\
#     --att_se_nonlinear relu\
#     --rho 0.5 \
#     --adver_type act \
#     --att_prior_type 'contextual' \
#     --alpha_gamma 1.0 \
#     --att_kl 0.01 \
#
# done


#   --seed 1 \
  # --per_device_train_batch_size 8 \
  # --per_device_eval_batch_size 8 \
  # --gradient_accumulation_steps 1\
# --per_device_train_batch_size 32 \

#

GLUE_DIR=../../../../../../huggingface/GLUE
TASK_NAME=MRPC
ID=Small_test
RES_LOC=../../../../../results/GLUE/$TASK_NAME/ALBERT

python run_glue.py \
  --local_rank -1 \
  --seed 1 \
  --model_name_or_path albert-base-v2 \
  --task_name  $TASK_NAME \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1\
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --overwrite_output_dir \
  --output_dir ../../../../../original_results/GLUE/$TASK_NAME/ALBERT/$ID/ \
  --att_se_hid_size 16\
  --att_se_nonlinear relu\
  --rho 0.5 \
  --adver_type act \
  --att_prior_type 'contextual' \
  --alpha_gamma 1.0 \
  --att_kl 0.01 \

# --max_steps 800 \
# --warmup_steps 200\

# --data_dir $GLUE_DIR/$TASK_NAME \
# --max_seq_length 128 \

# --per_device_train_batch_size 32 \

# python3 run_glue.py \
#   --local_rank -1 \
#   --seed 1 \
#   --model_type albert \
#   --model_name_or_path albert-base-v2 \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --data_dir $GLUE_DIR/$TASK_NAME \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 8 \
#   --per_gpu_eval_batch_size 8 \
#   --gradient_accumulation_steps 1\
#   --learning_rate 2e-5 \
#   --max_steps 800 \
#   --warmup_steps 200\
#   --doc_stride 128 \
#   --num_train_epochs 3.0 \
#   --save_steps 9999\
#   --output_dir ./results/GLUE/$TASK_NAME/ALBERT/$ID/ \
#   --do_lower_case \
#   --overwrite_output_dir \
#   --label_noise 0.2\
#   --att_kl 0.01\
#   --att_se_hid_size 16\
#   --att_se_nonlinear relu\
#   --att_type soft_attention \
#   --adver_type act \
#   --rho 0.5 \
#   --model_type whai \
#   --prior_gamma 2.70 \
#   --three_initial 0.0 \
#   --att_prior_type 'contextual' ;
