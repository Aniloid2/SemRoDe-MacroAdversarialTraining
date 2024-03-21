# BERT and ROBERTA Codebase

This codebase is designed for fine-tuning pre-trained BERT and ROBERTA models on MR, AG NEWS, and SST2 datasets. 

## Quick Start

### Workspace

- **Main Workspace Directory:** `/adv/`
- **Main Python File:** `/adv/high_level_at.py`
- **Bash Scripts for Testing:** `/adv/bash_scripts/`

### Dependencies

One crucial modification we made is to the `trainer.py` file found in:

```
adv/training_utils/trainer_branch.py
```

To accommodate some changes, you'll need to install our version of TextAttack and TextDefender locally:

```bash
pip install -e /src/TextAttack/
pip install -e /src/TextDefender/
```

In case you want to run the DSRM baseline you'll also need to install higher locally

```bash
pip install -e /higher
```

#### Environment

- **Python Version:** 3.8.5
- **Conda Environment Export:** `/adv/MAT.yaml`

We also have the individual requirement list in `requirements.txt`



### Running Tests

A straightforward example test is as follows:

```bash
CUDA_VISIBLE_DEVICES=5  python ./adv/high_level_at.py --attack_train TextFooler --attack_evaluate TextFooler --dataset 'MR' --model 'BERT' --method 'MMD' --method_type None --method_val 1 --save_space 'mmd_test' --GPU '1.1' --frozen 'True' --eval_method 'epoch' --epochs 7 --online_epochs 7 --batch_size 64 --data_ratio 0.1  --debug;
```

This command will initiate the MMD with counter-fitted embeddings, where only 10% (`--data_ratio 0.1`) of the dataset is utilized to generate the adversarial dataset. The generated adversarial data will be saved in `/adv/caches/`.

The process includes training for 7 epochs on the BERT model using the MR dataset. After training, the saved weights from each epoch are evaluated using the TextFooler attack. To adjust the evaluation to occur only at the last epoch, change `--eval_method 'epoch'` to `--eval_method 'last'`.

### Results

- The results are stored in: `/adv/LOCAL_Test/GLUE/MR/BERT/MMD_Test`
- If `--eval_method 'last'` is used, performance metrics by TextFooler for the last epoch are stored in the file `R.txt`.
- If `--eval_method 'epoch'` is selected, you'll find the results stored as `E1.txt`, `E2.txt`, etc., representing each epoch.


### Extra notes
Most tests were conducted on 1 Nvidia V100 with 32GB of memory. 

### TODO
Hyper parameter to increase, decrease parallelism, at the moment it's set up to generate the adversarial samples on 1 GPU with 8 model instances in parallel, some users may run out of memory.