def get_evaluation_freq(Method,e):
    if Method == 'last':
        method = 'last_model'
        name_file = 'R.txt'
    elif Method == 'best':
        method = 'best_model'
        name_file = 'B.txt'
    elif Method == 'epoch':
        method = f'checkpoint-epoch-{e}'
        name_file = f'E{e}.txt'
    return method,name_file

def get_attack(Attack=None,model_wrapper=None,cos_sim=0.5,sem_sim=0.5,no_cand=50,max_modification_rate = None):

    if Attack == 'TextFooler':
        development_attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'PWWS':
        development_attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper,max_modification_rate = max_modification_rate,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'BERTAttack':
        development_attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper,max_modification_rate = max_modification_rate,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'TextBugger':
        development_attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper,max_modification_rate = max_modification_rate,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'A2T':
        development_attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper,max_modification_rate = max_modification_rate,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'HotFlip':
        development_attack = textattack.attack_recipes.HotFlipEbrahimi2017.build(model_wrapper,max_modification_rate = max_modification_rate,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'Micro':
        development_attack = textattack.attack_recipes.MicroBrian2023.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'CharInsertion':
        development_attack = textattack.attack_recipes.CharInsertion.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'PuncInsertion':
        development_attack = textattack.attack_recipes.PuncInsertion.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'TextFoolerGS':
        development_attack = textattack.attack_recipes.TextFoolerGS.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack ==  'TextFoolerBeamSearch':
        development_attack = textattack.attack_recipes.TextFoolerBeamSearch.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack ==  'TextFoolerPSO':
        development_attack = textattack.attack_recipes.TextFoolerPSO.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    


    return development_attack

def get_eval_dataset(Dataset=None,model_wrapper=None, max_length = None ):

    # we always use the first 250 and last 250 samples of a dataset to ensure good class diversity
    if Dataset == 'MR':
       
       
        dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test')
        
        
        dataset.filter_dataset_by_sample_lenght(min_sample_lenght = 4)
        
        
        c0 = 0
        c1 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('Test Distribution','class-0',c0,  'class-1',c1)
    elif Dataset == 'AGNEWS':
        dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'test[0:250]+test[-250:]') 
        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
            elif j[-1] == 2:
                c2+=1
            elif j[-1] == 3:
                c3+=1
        print ('Test distribution','class-0:',c0,'class-1:' , c1, 'class-2:' ,c2,'class-3:',c3)
    elif Dataset == 'SST2': 
        dataset = textattack.datasets.HuggingFaceDataset('glue', 'sst2','validation[250:500]+validation[-500:-250]') 
        c0 = 0
        c1 = 0 
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1 
        print ('Test distribution','class-0:',c0,'class-1:' , c1,  )


    return dataset


def get_train_dataset(Dataset=None, Dataset_type='Generative',model_wrapper=None, max_length = None):
    # we use all train data
    # we also load any data that is not the first 250 or last 250 samples for evaluation purposes
    # these extra samples are not used for training but just to calculate the adv loss, OT, MMD loss etc.
    
    if Dataset == 'MR': 
        train_dataset, eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train')._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
        
        
        train_dataset = textattack.datasets.HuggingFaceDataset(train_dataset, None,'train')
        eval_dataset = textattack.datasets.HuggingFaceDataset(eval_dataset, None,'eval')
 
        
        c0 = 0
        c1 = 0
        for i,j in enumerate(train_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('Train Distribution','class-0:',c0 , 'class-1:',c1)
        
        
    elif Dataset == 'AGNEWS':
        train_dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'train')
        


        eval_dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'test[250:500]+test[-500:-250]')


        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        for i,j in enumerate(train_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
            elif j[-1] == 2:
                c2+=1
            elif j[-1] == 3:
                c3+=1
        print ('Train Distribution','class-0',c0,'class-1' , c1, 'class-2' ,c2,'class-3',c3)
    elif Dataset == 'SST2': 

        train_dataset, eval_dataset = textattack.datasets.HuggingFaceDataset('glue', 'sst2','train')._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
            
            
        train_dataset = textattack.datasets.HuggingFaceDataset(train_dataset, None,'train')
        eval_dataset = textattack.datasets.HuggingFaceDataset(eval_dataset, None,'eval')
 
        c0 = 0
        c1 = 0 
        for i,j in enumerate(train_dataset):
            # print (i,j) 

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1 
        
        print ('Train distribution','class-0:',c0,'class-1:' , c1 ) 
    
    return train_dataset,eval_dataset





def get_model(args=None,origin_folder=None,Model_save='BERT',Frozen = 'True',method_test='B',model_max_length=128):
 
    if Model_save == 'BERT':

        if method_test == 'ASCC':
            from transformers import BertConfig 

            if args.dataset == 'AGNEWS':
                num_labels = 4
            else: 
                num_labels = 2 
            model = local_models.ASCCBertModel.from_pretrained(origin_folder,num_labels=4 )  
            
            vocab = model.tokenizer.get_vocab()

            alpha = 10
            num_steps = 5
            current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) )
             
            nbr_file = os.path.join(current_file_path, 'requirements/counterfitted_neighbors.json')

            save_nbr_file = os.path.join(current_file_path, f'requirements/nbr_file_{Model_save}.pt')

            print ('Building neighbors from counterfitted_neighbors.json. Will take some time')
            model.build_nbrs( nbr_file = nbr_file , vocab=vocab, alpha=alpha, num_steps=num_steps,save_nbr_file=save_nbr_file) 
        elif method_test == 'DSRM':
            from transformers import BertConfig
            textattack.shared.utils.set_seed(765)
            if args.dataset == 'AGNEWS':
                num_labels = 4
            else: 
                num_labels = 2
            config = BertConfig.from_pretrained(origin_folder,num_labels=num_labels)


            if origin_folder == 'bert-base-uncased':
                model = local_models.BertForSequenceClassificationDSRM(args, origin_folder, config)
            else:
                model = local_models.BertForSequenceClassificationDSRM(args, 'bert-base-uncased', config)
                state_dict = torch.load(os.path.join(origin_folder, 'pytorch_model.bin'))
                model.load_state_dict(state_dict)
            

            
            
            
        else:
            # it's the same as a normal bert model but returns extra attributes through a custom class such as the pooling layers
            model = local_models.BertForSequenceClassificationOT.from_pretrained(origin_folder)



        attack_bool = {"attack":True}
        model.config.update(attack_bool)

    elif Model_save == 'ROBERTA':

        if method_test == 'ASCC':

            from transformers import BertConfig
            model = local_models.ASCCRobertaModel.from_pretrained(origin_folder)
            vocab = model.tokenizer.get_vocab()

            alpha = 10
            num_steps = 5
            current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) )
             
            nbr_file = os.path.join(current_file_path, 'requirements/counterfitted_neighbors.json')

            save_nbr_file = os.path.join(current_file_path, f'requirements/nbr_file_{Model_save}.pt')

            print ('Building neighbors from counterfitted_neighbors.json. Will take some time')
            model.build_nbrs( nbr_file = nbr_file , vocab=vocab, alpha=alpha, num_steps=num_steps,save_nbr_file=save_nbr_file) 
        
        elif method_test == 'DSRM':
            from transformers import RobertaConfig
            textattack.shared.utils.set_seed(765)
            if args.dataset == 'AGNEWS':
                num_labels = 4
            else: 
                num_labels = 2
            config = RobertaConfig.from_pretrained(origin_folder,num_labels=num_labels)

            if origin_folder == 'roberta-base':
                model = local_models.RobertaForSequenceClassificationDSRM(args, origin_folder, config)
            else:
                model = local_models.RobertaForSequenceClassificationDSRM(args, 'roberta-base', config)
                state_dict = torch.load(os.path.join(origin_folder, 'pytorch_model.bin'))
                model.load_state_dict(state_dict)   

            
        else:
            model = local_models.RobertaForSequenceClassificationOT.from_pretrained(origin_folder)
             
            
        attack_bool = {"attack":True}
        model.config.update(attack_bool)
        print ('Model Config',model.config)
    
 
    
    if origin_folder:
        tokenizer = transformers.AutoTokenizer.from_pretrained(origin_folder,model_max_length=model_max_length)
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    else:
        raise ValueError('No model has been passed to huggingface')


    if Frozen == 'True':
        for i in range(0,11):
            for name, param in model.named_parameters():
                if 'embeddings' in name:
                    param.requires_grad = False
                check = f'.{i}.'
                if check in name:
                    param.requires_grad = False
    else:
        pass
    return model_wrapper

def attack_model(output_folder,name_file,attack,dataset,query_budget=None,parallel=False,attack_num_workers_per_device=1):


    attack_args = textattack.AttackArgs(
        num_examples=500,
        log_to_txt=f"{output_folder}/{name_file}",
        parallel= parallel, 
        num_workers_per_device= attack_num_workers_per_device, 
        disable_stdout=True,
        silent=True,
        set_seed = False,)   


    attacker = textattack.Attacker(attack, dataset, attack_args) 
    r = attacker.attack_dataset()

    return r



if __name__ == '__main__':
    import transformers
    from textattack.models.wrappers import HuggingFaceModelWrapper
    import textattack

    import training_utils

    from datasets import load_dataset
    import datasets
    import argparse
    import torch
    import os
    import shutil
    import sys
    import warnings
    from textattack.transformations import WordSwapEmbedding
    import local_models
    import numpy as np
    import shutil
    import glob
    import pandas as pd
    import math
    # sys.path.append("/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv")

    # origin_folder = "./GLUE/MR/BERT/OT_GL_DMR_AKTTextFooler_ATETextFooler_B1_0_AT0_0_OT0_01_MMD0_0_CRL0_0_DR0_05_E12_LR2e-05_LRP2e-06_NWS0_FT/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MR", type=str, help="The dataset to use")
    parser.add_argument("--model", default="BERT", type=str, help="The model to use")
  
    parser.add_argument("--method", default="OT", type=str, help="method can be B, MMD, CRL, OT , AT, FLB or Embedding")
    parser.add_argument("--method_type", default="L2", type=str, help="only used when method == Embedding to specify the embedding technique type DSRM, INFOBERT")
    parser.add_argument("--method_val", default=1, type=float, help="The lambda value associated with method for MMD,CRL,OT is how much the regularizer influences training")
    
    parser.add_argument("--eval_method", default="epoch", type=str, help="eval every checkpoint (epoch), eval the best checkpoint (best), eval last (last)")
    parser.add_argument("--epochs", default=7, type=int, help="how many training epochs? default 7 for MR, 3 for AGNEWS and SST2")
    parser.add_argument("--online_epochs", default=-1, type=int, help="every how many epochs do we want to generate a new set of adversarial examples for SemRoDe (OT,MMD,CRL) make online_epochs=epochs to generate only 1 time at the start")
    
    parser.add_argument("--batch_size", default=64, type=int, help="batch size to train the model on")
    parser.add_argument("--data_ratio", default=0.1, type=float, help="the ammount of adversarial data as a ratio of original clean data, in this default case it's 10%")
    parser.add_argument("--train", default=True, type=lambda x: x.lower() == "true", nargs='?', const=True, help="do you want to train or only evaluate, set train=False if you arlready have checkpoints")
    parser.add_argument("--save_space", default="MMD_Main_Test", type=str, help="where to save the tests")
    parser.add_argument("--frozen", default="True", type=str, help="Do we train all of bert or only the last layer?")
    parser.add_argument("--attack_train", default="TextFooler", type=str, help="What attack to use to train the system")
    parser.add_argument("--attack_evaluate", default="TextFooler", type=str, help="What attack to use to evaluate the system")
    parser.add_argument("--modification_rate", default='ST', type=str, help="size of modification rate on adv dataset generation, default is a maximum modification rate of 30% (not sure it works on other settings anymore)")
    parser.add_argument('--debug', action='store_true', default=False, help='Turn on debug mode (outdated param now its always in debug)')
    parser.add_argument("--GPU", default="0.0", type=str, help="GPU currently under use (outdated param)")
    args = parser.parse_args()
    
    Dataset = args.dataset 
    Model =  args.model
    parallel = False
    attack_num_workers_per_device = 1
    archive_dir = args.save_space


    current_file_path = os.path.abspath(__file__) 
    current_file_dir = os.path.dirname(current_file_path) 
    parent_dir = os.path.dirname(current_file_dir)
    
    archive_destination = f"{current_file_dir}/LOCAL_Test/GLUE/{Dataset}/{Model}/{archive_dir}/"

    production_folder = f'{current_file_dir}/GLUE/{Dataset}/{Model}/'

    cache_path = f'{current_file_dir}'


    Id = 'OT_GL'

    Dataset_type = 'Normal'

    AttackTrain = args.attack_train
    
    AttackEvaluate = args.attack_evaluate

    Train = args.train 
    
    Evaluate_Attack = True


    Checkpoint = 1
    global_batch_size = args.batch_size
    accumulation_steps = 1
    NWS = 0
    WD = 0
    
    
    attack_method = {'BERTAttack':50, 'TextFooler':50, 'PWWS':50, 'TextBugger':50, 'A2T':50,'HotFlip':50, 'Micro':50,'CharInsertion':50, 'Morphin':50, 'PuncInsertion':50,'TextFoolerGS':50,'TextFoolerMaxLoss':50, 'TextFoolerGenetic':50,'TextFoolerAlzantot':50,'TextFoolerBeamSearch':50,'TextFoolerPSO':50}


    # used for experimenting
    # changes the train and eval semantic similarity threshold
    SS = [{'train':0.5,'eval':0.5}]
   
   # changes the cosine similarity 
    CS = [{'train':0.5,'eval':0.5}]

    # changes the number of embeddings
    NC =  [{'train':50,'eval':50}]



    # query budget is proportional to the number of tokens in the sample, we implemented this similar to textdefender
    query_budget = 50 

    # how much do we want to modify the sample e.g word substitution modifiation % (max)
    modification_rates_atk = {'AGNEWS':0.3,'MR':0.3,'YELP':0.3,"MNLI":0.3,'IMDB':0.1,'SST2':0.3}
    if args.modification_rate == 'ST':
        modification_rates_adv = {'AGNEWS':0.3,'MR':0.3,'YELP':0.3,"MNLI":0.3,'IMDB':0.1,'SST2':0.3}
    elif args.modification_rate == 'None':
        modification_rates_adv = {'AGNEWS':None,'MR':None,'YELP':None,"MNLI":None,'IMDB':None,'SST2':None}
    elif (float(args.modification_rate) <= 1) and (float(args.modification_rate) >= 0) :
        modification_rate = float(args.modification_rate) 
        modification_rates_adv = {'AGNEWS':modification_rate,'MR':modification_rate,'YELP':modification_rate,"MNLI":modification_rate,'SST2':modification_rate}
    else:
        ValueError('Modification rates issue')
    

    max_modification_rate_atk = modification_rates_atk[Dataset]
    max_modification_rate_adv = modification_rates_adv[Dataset]
    


    # max input context lenght, we keep it at 128 during testing for speed purposes.
    model_max_lengths = {'AGNEWS':128,'MR':128,'YELP':128, "MNLI":128,"IMDB":128,'SST2':128}
    model_max_length = model_max_lengths[Dataset]

    Data_ratios = [args.data_ratio]

    if Dataset == 'MR' and Model == 'BERT':
        if args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            else:
                pure_origin_model = "textattack/bert-base-uncased-rotten-tomatoes"
        else:
            pure_origin_model = "textattack/bert-base-uncased-rotten-tomatoes"
         
        parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        if args.online_epochs == -1:
            # generate adv attacks only one time at the start of the training
            AEI = [args.epochs]
        else:
            # create adv samples at the first epoch and after every online_epochs epochs.
            AEI = [args.online_epochs]
        Data_ratios = [args.data_ratio]
    elif Dataset == 'AGNEWS' and Model == 'BERT':
        # pure_origin_model = 'textattack/bert-base-uncased-ag-news'
        if args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            else:
                pure_origin_model = 'textattack/bert-base-uncased-ag-news'
        else:
            pure_origin_model = 'textattack/bert-base-uncased-ag-news'
        parallel = False
        attack_num_workers_per_device = 1
        Epochs = [args.epochs]
        if args.online_epochs == -1:
            # generate adv attacks only one time at the start of the training
            AEI = [args.epochs]
        else:
            # create adv samples at the first epoch and after every online_epochs epochs.
            AEI = [args.online_epochs]
        Data_ratios = [args.data_ratio]
    elif Dataset == 'SST2' and Model == 'BERT':
        if args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            else:
                pure_origin_model = "textattack/bert-base-uncased-SST-2"
        else:
            pure_origin_model = "textattack/bert-base-uncased-SST-2"
        parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        if args.online_epochs == -1:
            # generate adv attacks only one time at the start of the training
            AEI = [args.epochs]
        else:
            # create adv samples at the first epoch and after every online_epochs epochs.
            AEI = [args.online_epochs]
        Data_ratios = [args.data_ratio] 
    elif Dataset == 'MR' and Model == 'ROBERTA':
        # we furst have to train our own model from scratch before we finetune this on any alignment technique 
        # this is because we slighltly change the classification head of bert sequence classification
        
        if args.method == 'B': # for roberta, we slightly change the classification head to be 1 layer. this requires finetuning on the dataset, i then use this new model when training for robustness
            pure_origin_model = "textattack/roberta-base-rotten-tomatoes"

        elif args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            else:
                pure_origin_model = f"{current_file_dir}/LOCAL_Test/GLUE/MR/ROBERTA/NAACL_Baseline/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-7"
        else:
            pure_origin_model = f"{current_file_dir}/LOCAL_Test/GLUE/MR/ROBERTA/NAACL_Baseline/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-7"

        
        parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        if args.online_epochs == -1:
            # generate adv attacks only one time at the start of the training
            AEI = [args.epochs]
        else:
            # create adv samples at the first epoch and after every online_epochs epochs.
            AEI = [args.online_epochs]
        Data_ratios = [args.data_ratio]
    elif Dataset == 'AGNEWS' and Model == 'ROBERTA':
        # we furst have to train our own model from scratch before we finetune this on any alignment technique 
        # this is because we slighltly change the classification head of bert sequence classification
        if args.method == 'B':
            pure_origin_model = 'textattack/roberta-base-ag-news'


        elif args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            else:
                pure_origin_model =  f'/{current_file_dir}/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        else:
            pure_origin_model = f'/{current_file_dir}/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        
        parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        if args.online_epochs == -1:
            # generate adv attacks only one time at the start of the training
            AEI = [args.epochs]
        else:
            # create adv samples at the first epoch and after every online_epochs epochs.
            AEI = [args.online_epochs]
        Data_ratios = [args.data_ratio]
    elif Dataset == 'SST2' and Model == 'ROBERTA':
        # we furst have to train our own model from scratch before we finetune this on any alignment technique 
        # this is because we slighltly change the classification head of bert sequence classification
        
        if args.method == 'B':
            pure_origin_model = "textattack/roberta-base-SST-2"

        elif args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            else:
                pure_origin_model = f'/{current_file_dir}/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        
        else:
            pure_origin_model = f'/{current_file_dir}/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        
        parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        if args.online_epochs == -1:
            # generate adv attacks only one time at the start of the training
            AEI = [args.epochs]
        else:
            # create adv samples at the first epoch and after every online_epochs epochs.
            AEI = [args.online_epochs]
        Data_ratios = [args.data_ratio]



  



    Frozen = args.frozen 
    

    Eval_method = args.eval_method
    

    dic_methods = { 'MMD':'MMD', 'CRL':'CRL', 'AT':'AT', 'OT':'OT','B':'B','AUG':'AUG','MMD_AT':'MMD_AT','MMD_P':'MMD_P','P':'P','PF':'PF','MMD_PF':'MMD_PF','FLB':'FLB','MMD_FLB':'MMD_FLB','Online_AT':'Online_AT','Dist':'Dist', 'Embedding':'Embedding'}
    method_test = dic_methods[args.method]





    if method_test == 'OT':
        

        if Dataset == 'MR':
            Lambdas = [{'B':1.0,'AT':0.0,'OT':0.001,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
            Learning_rate= [[0.01]]
        elif Dataset == 'AGNEWS':
            Lambdas = [{'B':1.0,'AT':0.0,'OT':1.0,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
            Learning_rate = [[0.00002]]
        elif Dataset == 'SST2':
            Lambdas = [{'B':1.0,'AT':0.0,'OT':1.0,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
            Learning_rate = [[0.00002]]


    elif method_test == 'CRL': 

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.,'MMD':0.,'CRL':1.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]


    elif method_test == 'B':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.,'MMD':0.,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]

    elif method_test == 'AT': 
        Lambdas = [{'B':1.0,'AT':1.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'MMD':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'Dist':

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':args.method_type,'Dist_Val':args.method_val,'Online_AT':None}]
        if Dataset == 'MR':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'Embedding':

        Lambdas = [{'B':0.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Embedding':args.method_type,'Embedding_Val':args.method_val,'Online_AT':None}]
        if Dataset == 'MR':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]

    elif method_test == 'AUG':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':'Adv','P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]

    elif method_test == 'FLB':
        Lambdas = [{'B':0.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'Dist':None,'Dist_Val':0,'FLB':1.0,'FLB_Val':1.0,'Online_AT':None} ]

        if Dataset == 'MR':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]







    for i,LR in enumerate(Learning_rate):
        for s,sem_sim in enumerate(SS):
            for c,cos_sim in enumerate(CS):
                for nc,no_cand in enumerate(NC):
                    for j,Lda in enumerate(Lambdas):
                        for k,Data_ratio in enumerate(Data_ratios):
                            origin_folder = pure_origin_model 




                            for ep,epoch in enumerate(Epochs):

                                # create another T_Name_Wildcard, to check if Att_Train_MR_max_mod_r where ATK train is TextFooler?
                                 
                                print ('method',args.method,args.method_type,args.method_val,Lda[f'{args.method}'])
                                if args.method == 'Embedding':
                                    T_Name = f"{Id}_D{Dataset}_AKT{AttackTrain}_MR{max_modification_rate_adv}_ATE{AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_AUG{str(Lda['AUG'])}_{args.method}{Lda[f'{args.method}']}_{args.method}_Val{Lda[f'{args.method}_Val']}_FLB{Lda['FLB']}_OAT_{Lda['Online_AT']}_DR{Data_ratio}_E{sum(Epochs[:ep+1])}_LR{LR[0]}_SS{sem_sim['train']}_{sem_sim['eval']}_CS{cos_sim['train']}_{cos_sim['eval']}_NC{no_cand['train']}_{no_cand['eval']}_NWS{NWS}_F{Frozen[0]}"
                                else:
                                    T_Name = f"{Id}_D{Dataset}_AKT{AttackTrain}_MR{max_modification_rate_adv}_ATE{AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_AUG{str(Lda['AUG'])}_Dist{Lda[f'{args.method}']}_Dist_Val{Lda[f'{args.method}']}_FLB{Lda['FLB']}_OAT_{Lda['Online_AT']}_DR{Data_ratio}_E{sum(Epochs[:ep+1])}_LR{LR[0]}_SS{sem_sim['train']}_{sem_sim['eval']}_CS{cos_sim['train']}_{cos_sim['eval']}_NC{no_cand['train']}_{no_cand['eval']}_NWS{NWS}_F{Frozen[0]}"
                                T_Name = T_Name.replace('.','_')

                                output_folder = production_folder + T_Name


                                 
                                for Wildcard_AttackEvaluate in attack_method.keys():
                                    if args.method == 'Embedding':
                                        T_Name_Wildcard = f"{Id}_D{Dataset}_AKT{AttackTrain}_MR{max_modification_rate_adv}_ATE{Wildcard_AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_AUG{str(Lda['AUG'])}_{args.method}{Lda[f'{args.method}']}_{args.method}_Val{Lda[f'{args.method}_Val']}_FLB{Lda['FLB']}_OAT_{Lda['Online_AT']}_DR{Data_ratio}_E{sum(Epochs[:ep+1])}_LR{LR[0]}_SS{sem_sim['train']}_{sem_sim['eval']}_CS{cos_sim['train']}_{cos_sim['eval']}_NC{no_cand['train']}_{no_cand['eval']}_NWS{NWS}_F{Frozen[0]}"
                                    else:
                                        T_Name_Wildcard = f"{Id}_D{Dataset}_AKT{AttackTrain}_MR{max_modification_rate_adv}_ATE{Wildcard_AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_AUG{str(Lda['AUG'])}_Dist{Lda[f'{args.method}']}_Dist_Val{Lda[f'{args.method}']}_FLB{Lda['FLB']}_OAT_{Lda['Online_AT']}_DR{Data_ratio}_E{sum(Epochs[:ep+1])}_LR{LR[0]}_SS{sem_sim['train']}_{sem_sim['eval']}_CS{cos_sim['train']}_{cos_sim['eval']}_NC{no_cand['train']}_{no_cand['eval']}_NWS{NWS}_F{Frozen[0]}"
                                    T_Name_Wildcard = T_Name_Wildcard.replace('.','_')
                                    wild_card_path = archive_destination  + T_Name_Wildcard 
                                    
                                    if os.path.isdir( wild_card_path ): 
                                        
                                        clean_path = archive_destination  + T_Name
                                        
                                        print (f'Setting Train to False, Only evaluation is required')
                                        if os.path.isdir( clean_path ): 
                                            print ('Target Directory TK{AttackTrain}_ATE{AttackEvaluate} already exists, no need to copy, manualy delete this folder if checkpoints do not exist')
                                        else:
                                            Train = False
                                            
                                            print (f'Model trained on AKT{AttackTrain} and MR{max_modification_rate_adv} already exits')
                                            print (f'Copying ATK{AttackTrain}_ATE{Wildcard_AttackEvaluate} to ATK{AttackTrain}_ATE{AttackEvaluate}')
                                            print (f'Setting Train to False, Only evaluation is required')
                                            
                                            shutil.copytree(archive_destination + T_Name_Wildcard, archive_destination + T_Name) 
                                            try:
                                                for File in glob.glob(clean_path +'/E*'):  
                                                    os.remove( File)  
                                            except Exception as e:
                                                pass
                                            
                                            try:
                                                os.remove(clean_path + f'/R.txt')
                                            except Exception as e:
                                                pass
                                            try:
                                                os.remove(clean_path + f'/B.txt')
                                            except Exception as e:
                                                pass
                                            try:
                                                os.remove(clean_path + f'/P.txt')
                                            except Exception as e:
                                                pass
                                            try:
                                                os.remove(clean_path + f'/L.txt')
                                            except Exception as e:
                                                pass
                                            try:
                                                os.remove(clean_path + f'/L_Eval.txt')
                                            except Exception as e:
                                                pass
                                         
                                         
                                archive_exists = archive_destination + T_Name

                                if args.debug == False:
                                    if os.path.exists(archive_exists):

                                        break

                                output_folder = archive_exists

                                if args.debug == False:
                                    stout_dump_file = archive_exists +'/stdout.txt'
                                    sterr_dump_file = archive_exists +'/stderr.txt'
                                    os.makedirs(os.path.dirname(stout_dump_file), exist_ok=True)
                                    sys.stdout = open(stout_dump_file, 'w')
                                    os.makedirs(os.path.dirname(sterr_dump_file), exist_ok=True)
                                    sys.stderr = open(sterr_dump_file,'w')


                                if isinstance(Data_ratios[k],float):
                                    batch_size = global_batch_size
                                    if Data_ratios[k] < 0.1:
                                        ValueError('Data_ratios needs to have an entry that will lead to enought adv samples that can fill an entire batch, when working with small datasets, use a int as data_ratio e.g 5 sampels. this will change the size of the batches')
                                    pass
                                else:
                                    if Data_ratios[k] < global_batch_size:
                                        batch_size = Data_ratios[k]
                                        warnings.warn(f'changing batch size to {batch_size} since number of provided samples is smaller than current batch size')
                                    else:
                                        batch_size = global_batch_size


                                if Train:
                                    
                                    dump_file_name = f'{output_folder}/P.txt'

                                    os.makedirs(os.path.dirname(dump_file_name), exist_ok=True)
                                    file_object = open(dump_file_name,'a')
                                    parameter_dump = f"""Dataset:{Dataset}\nModel:{Model}
                                    \nID type:{Id}\nB_Val: {Lda['B']}\nAT_Val: {Lda['AT']}\nOT_Val:{Lda['OT']}\nMMD_Val:{Lda['MMD']}\nCRL_Val:{Lda['CRL']}\nAUG_Val:{Lda['AUG']}\n{args.method}:{Lda[f'{args.method}']}\nFLB_Val{Lda['FLB']}
                                    \nTraining Model: {pure_origin_model}
                                    \nData Ratio or number of adv samples:{Data_ratio}\nEpochs:{Epochs}
                                    \n(BS) Batch Size:{batch_size}\n(AS) Accumulation steps:{accumulation_steps}\n(WS) Warmup steps:{NWS}
                                    \n(LR1) Learning Rate1:{LR[:len(Epochs)]}\n(WD) Weight Decay:{WD}\n (MML) Model Max Length:{model_max_length}
                                    \n(Frozen) {Frozen}
                                    \n(Sem Sim Train) {sem_sim['train']} (Sem Sim Train) {sem_sim['eval']}
                                    \n(Cos Sim Train) {cos_sim['train']} (Cos Sim Train) {cos_sim['eval']}
                                    \n(Number Candidates Train) {no_cand['train']} (Number Candidates Train) {no_cand['eval']}
                                    \n(Query Budget) {query_budget}*L
                                    \n(Max Modification Rate ADV/ATK) ADV_{max_modification_rate_adv}/ATK_{max_modification_rate_atk}
                                    \n(AEI) Attack Epoch Interval:{AEI}
                                    \n(Attack) Attack type: Train ({AttackTrain}) Eval ({AttackEvaluate})
                                    \n(Parallel?) {parallel}
                                    \n(Num_workers) {attack_num_workers_per_device}
                                    \n(Eval Every) {Eval_method}

                                    \n(Method Type) {args.method_type}
                                    \n(Method Val) {args.method_val}

                                    \n(Debug?) {args.debug}"""
                                    print (parameter_dump)
                                    file_object.write(parameter_dump)
                                    file_object.close()




                                    model_wrapper = get_model(args,origin_folder,Model,Frozen=Frozen,method_test=args.method_type,model_max_length=model_max_length)
                                     
                                    train_dataset,eval_dataset = get_train_dataset(Dataset=Dataset,Dataset_type=Dataset_type,model_wrapper=model_wrapper, max_length= model_max_length )
                                     
                                    

                                    development_attack = get_attack(Attack=AttackTrain,model_wrapper=model_wrapper,cos_sim=cos_sim['train'],sem_sim = sem_sim['train'],no_cand=no_cand['train'],max_modification_rate = max_modification_rate_adv)
                                     

                                    training_args = training_utils.TrainingArgs(
                                                    method_test = method_test,
                                                    num_epochs=Epochs[ep],
                                                    model_name=Model,
                                                    num_clean_epochs=0, 
                                                    num_train_adv_examples=Data_ratio,
                                                    learning_rate=Learning_rate[i][ep],
                                                    per_device_train_batch_size=batch_size,
                                                    gradient_accumulation_steps=accumulation_steps,
                                                    num_warmup_steps=NWS,
                                                    weight_decay=WD,
                                                    attack_epoch_interval=AEI[ep],
                                                    log_to_csv=True,
                                                    log_to_tb = True,
                                                    csv_log_dir = f"{output_folder}/L.txt", 
                                                    tb_log_dir = f"{output_folder}/runs/",
                                                    output_dir=output_folder,
                                                    cache_path = cache_path,

                                                    B_Val = Lda['B'],
                                                    AT_Val = Lda['AT'],
                                                    OT_Val = Lda['OT'],
                                                    MMD_Val= Lda['MMD'],
                                                    CRL_Val= Lda['CRL'],
                                                    AUG_Val =  Lda['AUG'],
                                                    Dist = Lda['Dist'],
                                                    Dist_Val = Lda['Dist_Val'],
                                                    FLB_Val = Lda['FLB'],
                                                    Online_AT_Val =  Lda['Online_AT'],
                                                    Method_Dictionary = Lda,

                                                    Sem_Sim= sem_sim['train'],
                                                    Cos_Sim = cos_sim['train'],
                                                    No_Cand = no_cand['train'],
                                                    
                                                    query_budget_train = no_cand['train'], 
                                                    
                                                    max_modification_rate_adv = max_modification_rate_adv,

                                                    AttackTrain = AttackTrain,
                                                    AttackEvaluate = AttackEvaluate,

                                                    Data_ratio = Data_ratio,
                                                    Dataset_attack = Dataset,
                                                    checkpoint_interval_epochs=Checkpoint,
                                                    debug = args.debug,

                                                    attack_num_workers_per_device = attack_num_workers_per_device,
                                                    parallel = parallel
                                                )
                                    
                                    textattack.TrainingArgs = training_args


                                    trainer = training_utils.Trainer(
                                            model_wrapper,
                                            Id,
                                            development_attack,
                                            train_dataset,
                                            eval_dataset,
                                            training_args
                                        )

                                    textattack.Trainer = trainer
                                    
                                    
                                    trainer.train() 

                                    if not Evaluate_Attack:
                                        origin_folder = output_folder + '/checkpoint-epoch-{Epochs[ep]}' 
                                    else:
                                        origin_folder = output_folder # just set the origin folder to new test, we will load either per epoch,last or best
                                
                                


                                if Evaluate_Attack:


                                    if Train == False:
                                        origin_folder = output_folder

                                        parallel = False



                                    if Eval_method == 'epoch':
                                        
                                        _get_attack = True
                                        for e in range(1,Epochs[ep]+1):
                                            method, name_file = get_evaluation_freq(Eval_method,e)
                                            attack_folder = os.path.join(origin_folder,method)
                                            model_wrapper = get_model(args,attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length,method_test=args.method_type)
                                            model_wrapper.model.eval()
                                            model_wrapper.model.to(textattack.shared.utils.device)
                                            dataset = get_eval_dataset(Dataset=Dataset,model_wrapper=model_wrapper, max_length= model_max_length )

                                             
                                            if _get_attack:
                                                evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate_atk)
                                                _get_attack = False
                                            else:
                                                evaluation_attack.goal_function.model=model_wrapper
                                             
                                            # set seed and chose 500 samples randomly
                                            textattack.shared.utils.set_seed(765)
                                            indices = np.random.choice(len(dataset), 500, replace=False)
                                            dataset.filter_by_indices(indices)

                                            r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)

                                    elif Eval_method == 'last' or Eval_method == 'best':
                                        method, name_file = get_evaluation_freq(Eval_method,0)

                                        attack_folder = os.path.join(origin_folder,method)
                                        
                                        model_wrapper = get_model(args,attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length,method_test=args.method_type)
                                        model_wrapper.model.eval()
                                        model_wrapper.model.to(textattack.shared.utils.device)
                                        dataset = get_eval_dataset(Dataset=Dataset,model_wrapper=model_wrapper, max_length= model_max_length )

                                        evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate_atk)
                                        
                                        textattack.shared.utils.set_seed(765)
                                        indices = np.random.choice(len(dataset), 500, replace=False)

                                        dataset.filter_by_indices(indices)
                                        
                                        
                                        r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
 
                                    origin_folder = attack_folder

                                    model_wrapper.model.train()
                                    model_wrapper.model.to(textattack.shared.utils.device)
