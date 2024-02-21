GLUE_DIR=../../../huggingface/GLUE_SMALL/
TASK_NAME=MNLI
ID=OT
RES_LOC=../results/GLUE/$TASK_NAME/ALBERT
python3 ../run_glue.py \
  --local_rank -1 \
  --seed 42 \
  --model_type albert \
  --model_name_or_path $RES_LOC/$ID/ \
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --gradient_accumulation_steps 2\
  --learning_rate 3e-5 \
  --max_steps -1 \
  --warmup_steps 1000\
  --doc_stride 128 \
  --num_train_epochs 3.0 \
  --save_steps 9999\
  --output_dir ./results/GLUE/$TASK_NAME/ALBERT/$ID/New \
  --do_lower_case \
  --overwrite_output_dir \
  --label_noise 0.2\
  --att_kl 0.01\
  --att_se_hid_size 16\
  --att_se_nonlinear relu\
  --att_type soft_attention \
  --adver_type ot \
  --rho 0.5 \
  --model_type whai \
  --prior_gamma 2.70 \
  --three_initial 0.0
