
Our code works for BERT and ROBERTA. The codebase is compatible for finetuning pretrained BERT/ROBERTA model on MR, AGNEWS and SST2. We provide an example bash file finetune BERT. 

The main workspace is in /adv/

The main python file is /adv/high_level_at.py

we provide multiple bash files to run various tests in /adv/bash_scripts/

Since we heavily edit textattack (the trainer.py in /src/TextAttack/textattack/trainer.py) you would need to install our provided version of textattack from local

pip install -e /src/TextAttack/textattack

we run Python 3.8.5

we export our conda environment in /adv/MAT.yaml

a streight forward test:

CUDA_VISIBLE_DEVICES=0 python high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --save_space 'MMD_Test' --GPU '1.1' --frozen 'True' --epochs 7 --data_ratio 0.1 --eval_method 'epoch' --debug;

This will run MMD with counter-fitted embeddings where only 10% (--data_ratio 0.1) of the data is utalised to generated the adversarial dataset. The adversarial data will be saved to /adv/caches/

It will then train for 7 epochs on the BERT model and MR dataset. When finished training, we will run the TextFooler attack on the saved weights for each of the 10 epochs. change --eval_method 'epoch' to --eval_method 'last' to run evaluation only once at the last epoch.

The test files will be put in /adv/LOCAL_Test/GLUE/MR/BERT/MMD_Test

R.txt is the file that shows the TextFooler perofrmance on last epoch if --eval_method 'last'. Otherwise if 'epoch' E1.txt, E2.txt etc.