

LR=( 0.1 0.01 0.001 )
LRS=( "0_1" "0_01" "0_001" )
LR2=( 0.00002  0.00002  0.000002 )
LRS2=( "2e-5"  "2e-5"  "2e-6" )
for k in "${!LR[@]}"; do
   # test for LR 0.00005

  echo "${LR[k]}  ${LR2[k]}"
  CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py   --model  "textattack/bert-base-uncased-rotten-tomatoes" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --MMD_Val 0 --CRL_Val 0 --Data_ratio 0.05 --Epochs 50 --AEI 50 --BS 64 --AS 1 --NWS 0 --LR ${LR[k]} --WD 0.01 --Frozen "True" --Checkpoint 1 ;



  for i in "${LR[k]}" ; do for j in {1..50};
     do
       echo "W $i $j S"
      CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --MMD_Val 0 --CRL_Val 0 --Data_ratio 0.05 --Epochs 50 --E $j --NWS 0 --LR $i --Frozen "True" --Method 'epoch';
     done
   done


  CUDA_VISIBLE_DEVICES=2  python textattack_run_glue.py   --model  "./GLUE/MR/BERT/OT_GL_DMR_B1_0_AT0_0_OT0_01_MMD0_0_CRL0_0_DR0_05_E50_LR${LRS[k]}_NWS0_FT/checkpoint-epoch-50/" --Dataset 'MR' --id "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.05 --Epochs 10 --AEI 10 --BS 64 --AS 1 --NWS 0 --LR ${LR2[k]} --WD 0.01 --Frozen "True" --Checkpoint 1;
  # exit 1


   for i in  "${LR2[k]}" ; do for j in {1..10};
       do
         echo "W $i $j S"

         CUDA_VISIBLE_DEVICES=2 python attack_glue.py --model  "./GLUE/MR/BERT/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val 0.01 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.05 --Epochs 10 --E $j --NWS 0 --LR $i --Frozen "True" --Method 'epoch';

       done
    done

   mv ./GLUE/MR/BERT/OT* /nas/Optimal_transport/GLUE/MR/BERT/LR_OT_Test/
done
