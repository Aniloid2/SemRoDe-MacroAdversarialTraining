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
    # TODO: This code currently involves monkey patching of library classes, which
    # can lead to maintenance issues and unexpected behaviors. We need to refactor
    # our custom changes into separate classes or properly extend the existing classes
    # without altering the library code directly.
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
    elif Attack == 'Morphin':
        development_attack = textattack.attack_recipes.MorpheusTan2020.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'PuncInsertion':
        development_attack = textattack.attack_recipes.PuncInsertion.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'TextFoolerGS':
        development_attack = textattack.attack_recipes.TextFoolerGS.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'TextFoolerMaxLoss':
        development_attack = textattack.attack_recipes.TextFoolerMaxLoss.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack == 'TextFoolerGenetic':
        development_attack = textattack.attack_recipes.TextFoolerGenetic.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack ==  'TextFoolerAlzantot':
        development_attack = textattack.attack_recipes.TextFoolerAlzantot.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack ==  'TextFoolerBeamSearch':
        development_attack = textattack.attack_recipes.TextFoolerBeamSearch.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    elif Attack ==  'TextFoolerPSO':
        development_attack = textattack.attack_recipes.TextFoolerPSO.build(model_wrapper,max_modification_rate = max_modification_rate,cos_sim=cos_sim,sem_sim=sem_sim,no_cand=no_cand)
    


    return development_attack

def get_eval_dataset(Dataset=None,model_wrapper=None, max_length = None ):

    # we always use the first 250 and last 250 samples of a dataset to ensure good class diversity

    if Dataset == 'MRPC':
        dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test")




        dataset.filter_subset(counts_per_class = {0:167,1:333}) 

        c0 = 0
        c1 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('validate number of samples per class:','c0 , c1',c0,c1)

    elif Dataset == 'MNLI':
 
        dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched[0:250]+validation_matched[-250:]",None,{0: 1, 1: 2, 2: 0},shuffle=True)
        


 
        c0 = 0
        c1 = 0
        c2 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
            elif j[-1] == 2:
                c2+=1 
        print ('validate number of samples per class: ','c0',c0,'c1' , c1, 'c2' ,c2)

    elif Dataset == 'SNLI':

        eval_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        dataset.filter_by_labels_([0,1,2])
    elif Dataset == 'MR':
       
       
        dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test')
        
        
        dataset.filter_dataset_by_sample_lenght(min_sample_lenght = 4)
        
        
        c0 = 0
        c1 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('validate number of samples per class: ','c0 , c1',c0,c1)
    elif Dataset == 'IMDB':
        dataset = textattack.datasets.HuggingFaceDataset('imdb', None,'test[0:250]+test[-250:]')
        
        if model_wrapper and max_length :
            # new_train = train_dataset.process_dataset(max_length,model_wrapper.tokenizer)
            # train_df = pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])
            # train_table = datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data

            # new_train = datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data)
            
             
            dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'eval')
            # print('after',train_dataset[0])
            # train_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(train_dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'train')
            # eval_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(eval_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'eval')
              
            # eval_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(eval_dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'test')
             
        
        c0 = 0
        c1 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('c0 , c1',c0,c1)
    elif Dataset == 'AGNEWS':
        dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'test[0:250]+test[-250:]')
        # dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'test[0:5]+test[-5:]') 
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
        print ('test distribution','c0',c0,'c1' , c1, 'c2' ,c2,'c3',c3)
    elif Dataset == 'SST2':
        # dataset = textattack.datasets.HuggingFaceDataset('sst2', None,'test[0:250]+test[-250:]')
        dataset = textattack.datasets.HuggingFaceDataset('glue', 'sst2','validation[250:500]+validation[-500:-250]')
        # dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'test[0:5]+test[-5:]') 
        c0 = 0
        c1 = 0 
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1 
        print ('test distribution','c0',c0,'c1' , c1,  )

    elif Dataset == 'YELP':
        dataset = textattack.datasets.HuggingFaceDataset('yelp_polarity', None,'test[0:250]+test[-250:]')
        
        if model_wrapper and max_length :
            # new_train = datasets.Dataset(train_dataset.process_dataset(max_length,model_wrapper.tokenizer))
         
            # dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'test') 
            dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'eval')
              

        c0 = 0
        c1 = 0
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('test distribution','c0',c0,'c1' , c1)


    return dataset


def get_train_dataset(Dataset=None, Dataset_type='Generative',model_wrapper=None, max_length = None):
    # we use all train data
    # we also load any data that is not the first 250 or last 250 samples for evaluation purposes
    # these extra samples are not used for training but just to calculate the adv loss, OT, MMD loss etc.
    if Dataset == 'MRPC':
        # train_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="train[0:100]")
        # eval_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test[0:100]")
        train_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="train")
        eval_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test")

    elif Dataset == 'MNLI':
        train_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',split="train",dataset_columns=None,label_map={0: 1, 1: 2, 2: 0},shuffle=True)
        # eval_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched",None,{0: 1, 1: 2, 2: 0},shuffle=True)
        eval_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched[250:500]+validation_matched[-500:-250]",None,{0: 1, 1: 2, 2: 0},shuffle=True)

        # _,eval_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched")._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
        # train_dataset = textattack.datasets.HuggingFaceDataset(train_dataset, None,'train')
        # eval_dataset = textattack.datasets.HuggingFaceDataset(eval_dataset, None,'validation_matched')    

        c0 = 0
        c1 = 0
        c2 = 0
        for i,j in enumerate(eval_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
            elif j[-1] == 2:
                c2+=1
        print ('c0 , c1','c2',c0,c1,c2)
        # test[250:500]+test[-500:-250]
        

    elif Dataset == 'SNLI':
        train_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "train[0:10000]", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        train_dataset.filter_by_labels_([0,1,2])
        eval_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        eval_dataset.filter_by_labels_([0,1,2])
    elif Dataset == 'MR':

        if Dataset_type == 'Normal':
        
            train_dataset, eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train')._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
            
            
            train_dataset = textattack.datasets.HuggingFaceDataset(train_dataset, None,'train')
            eval_dataset = textattack.datasets.HuggingFaceDataset(eval_dataset, None,'eval')
 
        elif Dataset_type == 'Generative':
            sys.exit()
            mid_dataset = textattack.datasets.GenerativeDataset('rotten_tomatoes', None,'train')
            mid_dataset._add_prompts(dataset_type = Dataset) 
            train_dataset, eval_dataset = mid_dataset._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
            # train_dataset, eval_dataset = textattack.datasets.GenerativeDataset('rotten_tomatoes', None,'train')
            
            
            # ._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
            
            
            train_dataset = textattack.datasets.GenerativeDataset(train_dataset, None,'train')
            eval_dataset= textattack.datasets.GenerativeDataset(eval_dataset, None,'eval')
              
        # # old way of doing split
        # train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train').train_test_split(args.dev_ratio, seed=args.seed, shuffle=False).values()
        # # train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train[0:25]+train[-25:]')
        # # train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train[0:2]+train[-2:]')
        # train_dataset.shuffle()
        # # eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test') #test[249:-250]+test[-250:]
        # # eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:250]+test[-250:]')
        # # print ('len da',len(dataset),dataset[0],dataset[249],dataset[-249],dataset[-1])
        # eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[250:-250]')
        # # print (eval_dataset[0],eval_dataset[249],eval_dataset[-249],eval_dataset[-1])
        # # print ('len eval',len(eval_dataset))

        c0 = 0
        c1 = 0
        for i,j in enumerate(eval_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('c0 , c1',c0,c1)
        
        
        # eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:25]+train[-25:]')
    elif Dataset == 'IMDB':
        # train_dataset = textattack.datasets.HuggingFaceDataset('imdb', None,'train[:60000]+train[-60000:]')
        train_dataset = textattack.datasets.HuggingFaceDataset('imdb', None,'train')

        # print('before',train_dataset[0])

        eval_dataset = textattack.datasets.HuggingFaceDataset('imdb', None,'test[250:500]+test[-500:-250]')

        if model_wrapper and max_length :
            # new_train = train_dataset.process_dataset(max_length,model_wrapper.tokenizer)
            # train_df = pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])
            # train_table = datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data

            # new_train = datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data)
            
             
            train_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'train')
            # print('after',train_dataset[0])
            # train_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(train_dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'train')
            eval_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(eval_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'eval')
              
            # eval_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(eval_dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'test')
             



        # pMax = 0.1, Qmax = KmaxxL, semantic sim = 0.84
        # KMax = 50 (number of synonym sub), L lenght of input

        c0 = 0
        c1 = 0
        for i,j in enumerate(train_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('c0 , c1',c0,c1)
        
    elif Dataset == 'AGNEWS':
        train_dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'train')
        


        eval_dataset = textattack.datasets.HuggingFaceDataset('ag_news', None,'test[250:500]+test[-500:-250]')

        # pMax = 0.1, Qmax = KmaxxL, semantic sim = 0.84
        # KMax = 50 (number of synonym sub), L lenght of input



        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        for i,j in enumerate(eval_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
            elif j[-1] == 2:
                c2+=1
            elif j[-1] == 3:
                c3+=1
        print ('eval distribution','len eval',len(eval_dataset),'c0',c0,'c1' , c1, 'c2' ,c2,'c3',c3)
    elif Dataset == 'SST2':
        # train_dataset = textattack.datasets.HuggingFaceDataset('sst2', None,'train')
        # train_dataset = textattack.datasets.HuggingFaceDataset('glue', 'sst2','train')

        # eval_dataset = textattack.datasets.HuggingFaceDataset('sst2', None,'validation[250:500]+validation[-500:-250]')
        # eval_dataset = textattack.datasets.HuggingFaceDataset('glue', 'sst2','validation')
        # pMax = 0.1, Qmax = KmaxxL, semantic sim = 0.84
        # KMax = 50 (number of synonym sub), L lenght of input

        train_dataset, eval_dataset = textattack.datasets.HuggingFaceDataset('glue', 'sst2','train')._dataset.train_test_split(0.1, seed=42, shuffle=True).values()
            
            
        train_dataset = textattack.datasets.HuggingFaceDataset(train_dataset, None,'train')
        eval_dataset = textattack.datasets.HuggingFaceDataset(eval_dataset, None,'eval')

        print ('size dataset','train size:',len(train_dataset),'eval size', len(eval_dataset))

        c0 = 0
        c1 = 0 
        for i,j in enumerate(eval_dataset):
            # print (i,j) 

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1 
        
        print ('eval distribution','len eval',len(eval_dataset),'c0',c0,'c1' , c1,  ) 
    elif Dataset == 'YELP':
        train_dataset = textattack.datasets.HuggingFaceDataset('yelp_polarity', None,'train[:120000]')
        # train_dataset = textattack.datasets.HuggingFaceDataset('yelp_polarity', None,'train[:200]')
        # train_dataset = textattack.datasets.HuggingFaceDataset('yelp_polarity', None,'train[0:25]+train[-25:]')
        # train_dataset.shuffle()
        
        
        
        # eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test') #test[249:-250]+test[-250:]
        # dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:250]+test[-250:]')
        # print ('len da',len(dataset),dataset[0],dataset[249],dataset[-249],dataset[-1])
            
        eval_dataset = textattack.datasets.HuggingFaceDataset('yelp_polarity', None,'test[250:500]+test[-500:-250]')
        # print (eval_dataset[0],eval_dataset[249],eval_dataset[-249],eval_dataset[-1])
        # print ('len eval',len(eval_dataset))

        
        
        if model_wrapper and max_length :
            # new_train = train_dataset.process_dataset(max_length,model_wrapper.tokenizer)
            # train_df = pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])
            # train_table = datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data

            # new_train = datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data)
            
             
            train_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(train_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'train')
             
            # train_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(train_dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'train')
            eval_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(datasets.Dataset.from_pandas(pd.DataFrame(eval_dataset.process_dataset(max_length,model_wrapper.tokenizer), columns=['text', 'label'])).data),None,'eval')
              
            # eval_dataset = textattack.datasets.HuggingFaceDataset(datasets.Dataset(eval_dataset.process_dataset(max_length,model_wrapper.tokenizer)),None,'test')
             



        c0 = 0
        c1 = 0
        for i,j in enumerate(train_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('c0 , c1',c0,c1) 
    return train_dataset,eval_dataset





def get_model(args=None,origin_folder=None,Model_save='BERT',Frozen = 'True',method_test='B',model_max_length=128):
    # current_file_path = os.path.dirname(os.path.abspath(__file__)) 
    # parent_path =  os.path.dirname(os.path.dirname(current_file_path)) 
    # sys.path.append(f'{parent_path}/src/TextDefender') 
    # from utils.dne_utils import get_bert_vocab, get_roberta_vocab# WeightedEmbedding, DecayAlphaHull
    # VOCAB = {
    #             "BERT": get_bert_vocab,
    #             "ROBERTA": get_roberta_vocab,
    #             # "xlnet": get_xlnet_vocab,
    #             # "albert": get_albert_vocab,
    #         }
    print ('method_test',method_test)
    if Model_save == 'BERT':
        # if method_test == 'FLB':
        #     from transformers import BertConfig
        #     # config = BertConfig.from_pretrained(origin_folder)
        #     # config.output_hidden_states = True
        #     model = transformers.models.bert.BertForSequenceClassification.from_pretrained(origin_folder)
        if method_test == 'ASCC':
            from transformers import BertConfig 
            # config = BertConfig.from_pretrained(origin_folder,num_labels=num_labels)
            # print('origin_folder',origin_folder)
            if args.dataset == 'AGNEWS':
                num_labels = 4
            else: 
                num_labels = 2 
            model = local_models.ASCCBertModel.from_pretrained(origin_folder,num_labels=4 )  
            
            vocab = model.tokenizer.get_vocab()
            # print ('vocab',vocab)
            alpha = 10
            num_steps = 5
            current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) )
             
            nbr_file = os.path.join(current_file_path, 'requirements/counterfitted_neighbors.json')

            save_nbr_file = os.path.join(current_file_path, f'requirements/nbr_file_{Model_save}.pt')
            # print (current_file_path)
            # print ('nbr_file',nbr_file)
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

             
            # model = local_models.BertForSequenceClassificationDSRM(args,origin_folder,config)

            # model = local_models.BertForSequenceClassificationDSRM.from_pretrained(origin_folder)
            # sys.exit()
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



        

        # print ('model configuration',model.config)
    elif Model_save == 'ALBERT':
        model = transformers.models.albert.AlbertForSequenceClassificationOT.from_pretrained(origin_folder)
        attack_bool = {"attack":True}
        model.config.update(attack_bool)
        print ('model configuration',model.config)
        print ('need to be able to freeze the last 11 layers')
        sys.exit()
    elif Model_save == 'ROBERTA':
        # if method_test  == 'FLB':
            # model = transformers.models.roberta.RobertaForSequenceClassification.from_pretrained(origin_folder)
        if method_test == 'ASCC':
            # from transformers import BertConfig
            # model = local_models.ASCCModel.from_pretrained(origin_folder) 
            from transformers import BertConfig
            model = local_models.ASCCRobertaModel.from_pretrained(origin_folder)
            vocab = model.tokenizer.get_vocab()
            # print ('vocab',vocab)
            alpha = 10
            num_steps = 5
            current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) )
             
            nbr_file = os.path.join(current_file_path, 'requirements/counterfitted_neighbors.json')

            save_nbr_file = os.path.join(current_file_path, f'requirements/nbr_file_{Model_save}.pt')
            # print (current_file_path)
            # print ('nbr_file',nbr_file)
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

             
            # model = local_models.BertForSequenceClassificationDSRM(args,origin_folder,config)

            # model = local_models.BertForSequenceClassificationDSRM.from_pretrained(origin_folder)
            # sys.exit()
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
        print ('config',model.config)
    
    elif Model_save == 'GPTJ':
        
        model = local_models.GPTJForCausalLM.from_pretrained(origin_folder)
        
        
    
    if origin_folder:
        tokenizer = transformers.AutoTokenizer.from_pretrained(origin_folder,model_max_length=model_max_length)
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    else:
        raise ValueError('No model has been passed to huggingface')


    if Frozen == 'True':
        # print ('freezing', Frozen)
        for i in range(0,11):
            for name, param in model.named_parameters():
                if 'embeddings' in name:
                    param.requires_grad = False
                check = f'.{i}.'
                if check in name:
                    param.requires_grad = False
    else:
        # print ('dont freeze', Frozen)
        pass
    return model_wrapper

def attack_model(output_folder,name_file,attack,dataset,query_budget=None,parallel=False,attack_num_workers_per_device=1):

    # name_file,method = get_evaluation_freq(Method=Eval_method,e=e)


    attack_args = textattack.AttackArgs(
        num_examples=500,
        log_to_txt=f"{output_folder}/{name_file}",
        parallel= parallel, # True, #parallel,,
        num_workers_per_device= attack_num_workers_per_device, # 8, #attack_num_workers_per_device,
        # parallel=  parallel,
        # num_workers_per_device=  attack_num_workers_per_device,
        disable_stdout=True,
        silent=True,
        # query_budget = query_budget,
        set_seed = False,)   

    # attack_args = textattack.AttackArgs(
    #     num_examples=500,
    #     log_to_txt=f"{output_folder}/{name_file}",
    #     parallel= False, # True, #parallel,,
    #     num_workers_per_device= 1, # 8, #attack_num_workers_per_device,
    #     # parallel=  parallel,
    #     # num_workers_per_device=  attack_num_workers_per_device,
    #     disable_stdout=False,
    #     silent=False,
    #     # query_budget = query_budget,
    #     set_seed = False,

    # )   
 

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
    parser.add_argument("--method", default="OT", type=str, help="the method to use")
    parser.add_argument("--method_type", default="L2", type=str, help="the method to use")
    parser.add_argument("--method_val", default=1, type=float, help="the method to use")
    parser.add_argument("--eval_method", default="best", type=str, help="the method to use")
    parser.add_argument("--epochs", default=10, type=int, help="the method to use")
    parser.add_argument("--online_epochs", default=-1, type=int, help="after how many epochs do we generate new adv samples given the new current model")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size to train the model on")
    parser.add_argument("--data_ratio", default=0.1, type=float, help="the method to use")
    parser.add_argument("--train", default=True, type=lambda x: x.lower() == "true", nargs='?', const=True, help="the method to use")
    parser.add_argument("--save_space", default="OT_Weeklong_Test", type=str, help="where to save the tests")
    parser.add_argument("--frozen", default="True", type=str, help="Do we train a frozen bert or all bert?")
    parser.add_argument("--attack_train", default="TextFooler", type=str, help="What attack to use to train the system")
    parser.add_argument("--attack_evaluate", default="TextFooler", type=str, help="What attack to use to evaluate the system")
    parser.add_argument("--modification_rate", default='ST', type=str, help="size of modification rate on adv dataset generation")
    parser.add_argument('--debug', action='store_true', default=False, help='Turn on debug mode')
    parser.add_argument("--GPU", default="0.0", type=str, help="GPU currently under use")
    args = parser.parse_args()
    
    Dataset = args.dataset # high level in bash script we pass as arg

    Model =  args.model


    parallel = False
    attack_num_workers_per_device = 1

    # name_of_potential_dirs = {'OT_Weeklong_Test',CORAL_Test,MMD_Test}

    archive_dir = args.save_space
    # archive_destination = f"/nas/Optimal_transport/GLUE/{Dataset}/{Model}/{archive_dir}/"

    #temp solution
    archive_destination = f"/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/{Dataset}/{Model}/{archive_dir}/"


    Id = 'OT_GL'

    Dataset_type = 'Normal'

    production_folder = f'/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/GLUE/{Dataset}/{Model}/'

    cache_path = f'/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/'

    AttackTrain = args.attack_train
    # AttackEvaluate = 'TextFooler'
    AttackEvaluate = args.attack_evaluate

    Train = args.train 
    
    Evaluate_Attack = True


    Checkpoint = 1
    global_batch_size = args.batch_size# 64 16
    accumulation_steps = 1
    NWS = 0
    WD = 0
    Epochs = [10] # [50,10]

    AEI = [10] # [50,10]

    attack_method = {'BERTAttack':50, 'TextFooler':50, 'PWWS':50, 'TextBugger':50, 'A2T':50,'HotFlip':50, 'Micro':50,'CharInsertion':50, 'Morphin':50, 'PuncInsertion':50,'TextFoolerGS':50,'TextFoolerMaxLoss':50, 'TextFoolerGenetic':50,'TextFoolerAlzantot':50,'TextFoolerBeamSearch':50,'TextFoolerPSO':50}

    # SS = [{'train':0.5,'eval':0.1},{'train':0.5,'eval':0.2},{'train':0.5,'eval':0.3},{'train':0.5,'eval':0.4},{'train':0.5,'eval':0.5},{'train':0.5,'eval':0.6},{'train':0.5,'eval':0.7},{'train':0.5,'eval':0.8},{'train':0.5,'eval':0.9}] #[0.5,0.5]
    # SS = [{'train':0.5,'eval':math.pi/6},{'train':0.5,'eval':math.pi/3},{'train':0.5,'eval':math.pi/2},{'train':0.5,'eval':(2*math.pi)/3},{'train':0.5,'eval':(5*math.pi)/6},{'train':0.5,'eval':math.pi}] #[0.5,0.5]
    # SS = [{'train':0.5,'eval':math.pi/96},{'train':0.5,'eval':math.pi/12},{'train':0.5,'eval':math.pi/24},{'train':0.5,'eval':math.pi/48}]
    # SS = [{'train':0.5,'eval':math.pi/96}]
    SS = [{'train':0.5,'eval':0.5}]
    
    # SS = [0.1,0.3,0.4,0.6,0.7,0.8,0.9,0.5]
    CS = [{'train':0.5,'eval':0.5}]# [0.5,0.5]#,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.5]

    # SS = [0.11]
    # NC = [1,5,10,20,30,40,50,60,70,80,90,200,500]
    # NC =  [{'train':attack_method[AttackTrain],'eval': attack_method[AttackEvaluate]}] #[50,50]
    # if AttackTrain == 'BERTAttack':
    NC =  [{'train':50,'eval':50}]
    # NC =  [{'train':50,'eval': 1},{'train':50,'eval': 5},{'train':50,'eval': 10},{'train':50,'eval': 100},{'train':50,'eval': 150}] #[50,50]
    # NC =  [{'train':1,'eval': 50},{'train':5,'eval': 50},{'train':10,'eval': 50},{'train':100,'eval': 50},{'train':150,'eval': 50}] #[50,50]
    
    
    


    # nc = 0
    query_budget = attack_method[AttackTrain]

    modification_rates_atk = {'AGNEWS':0.3,'MR':0.3,'YELP':0.3,"MNLI":0.3,'IMDB':0.1,'SST2':0.3}
    
    if args.modification_rate == 'ST':
        modification_rates_adv = {'AGNEWS':0.3,'MR':0.3,'YELP':0.3,"MNLI":0.3,'IMDB':0.1,'SST2':0.3}
    elif args.modification_rate == 'None':
        modification_rates_adv = {'AGNEWS':None,'MR':None,'YELP':None,"MNLI":None,'IMDB':None,'SST2':None}
    elif (float(args.modification_rate) <= 1) and (float(args.modification_rate) >= 0) :
        modification_rate = float(args.modification_rate) 
        modification_rates_adv = {'AGNEWS':modification_rate,'MR':modification_rate,'YELP':modification_rate,"MNLI":modification_rate,'SST2':modification_rate}
    else:
        ValueError('modification rate issue')
    

    max_modification_rate_atk = modification_rates_atk[Dataset]
    max_modification_rate_adv = modification_rates_adv[Dataset]
    



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
        # pure_origin_model = "bert-base-uncased"
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger' or attack_method[AttackTrain] == 'HotFlip' :
            parallel = False
        else:
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
        # pure_origin_model = "bert-base-uncased"
    if Dataset == 'MNLI' and Model == 'BERT':
        pure_origin_model = "textattack/bert-base-uncased-MNLI"
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger':
            parallel = False
        else:
            parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        AEI = [args.epochs]
        Data_ratios = [args.data_ratio]
        # pure_origin_model = "bert-base-uncased"
    if Dataset == 'SST2' and Model == 'BERT':
        if args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "bert-base-uncased" # model has to be trained from scratch
            else:
                pure_origin_model = "textattack/bert-base-uncased-SST-2"
        else:
            pure_origin_model = "textattack/bert-base-uncased-SST-2"

        # pure_origin_model = "bert-base-uncased"
        # pure_origin_model = "textattack/bert-base-uncased-SST-2"
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger' or attack_method[AttackTrain] == 'HotFlip' :
            parallel = False
        else:
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
        # pure_origin_model = "bert-base-uncased"
    if Dataset == 'MRPC' and Model == 'BERT':
        pure_origin_model = "textattack/bert-base-uncased-MRPC"
    if Dataset == 'IMDB' and Model == 'BERT':

        # pure_origin_model = 'textattack/bert-base-uncased-imdb'
        if args.method == 'B':
            pure_origin_model = 'textattack/bert-base-uncased-imdb'
        else:
            pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/IMDB/BERT/Pooled_IMDB/OT_GL_DIMDB_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/checkpoint-epoch-3"
        
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger':
            parallel = False
        else:
            parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        AEI = [args.epochs]
    if Dataset == 'AGNEWS' and Model == 'BERT':
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
        AEI = [args.epochs]
        Data_ratios = [args.data_ratio]

        



    if Dataset == 'YELP' and Model == 'BERT':
        # pure_origin_model = 'textattack/bert-base-uncased-yelp-polarity'
        pure_origin_model =  "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/YELP/BERT/Pooled_YELP/OT_GL_DYELP_AKTTextFooler_MR0_3_ATETextFooler_B0_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_PF0_0_FLB1_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-3/"
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger':
            parallel = False
        else:
            parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        AEI = [args.epochs]
        Data_ratios = [args.data_ratio]
        # pure_origin_model = "bert-base-uncased"
    if Dataset == 'MR' and Model == 'ROBERTA':
        # pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/MR/ROBERTA/Pooled_MR/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/checkpoint-epoch-7/" #
        # "textattack/roberta-base-rotten-tomatoes"
        # if args.method == 'B':
        #     pure_origin_model = "textattack/roberta-base-rotten-tomatoes"
        # else:
        #     pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/MR/ROBERTA/NAACL_Baseline/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-7"
        

        if args.method == 'B': # for roberta, we slightly change the classification head to be 1 layer. this requires finetuning on the dataset, i then use this new model when training for robustness
            pure_origin_model = "textattack/roberta-base-rotten-tomatoes"
            # else:
            #     pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        elif args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            else:
                pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/MR/ROBERTA/NAACL_Baseline/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-7"
        else:
            pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/MR/ROBERTA/NAACL_Baseline/OT_GL_DMR_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E7_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-7"

        
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger' or attack_method[AttackTrain] == 'A2T' or attack_method[AttackEvaluate] == 'A2T':
            parallel = False
        else:
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
        # pure_origin_model = "bert-base-uncased"
    if Dataset == 'MRPC' and Model == 'ROBERTA':
        pure_origin_model = "textattack/roberta-base-uncased-MRPC"
    if Dataset == 'IMDB' and Model == 'ROBERTA':
        if args.method == 'B':
            pure_origin_model = 'aychang/roberta-base-imdb'
        else:
            pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/IMDB/ROBERTA/Pooled_IMDB/OT_GL_DIMDB_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/checkpoint-epoch-3"
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger':
            parallel = False
        else:
            parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        AEI = [args.epochs]
    if Dataset == 'AGNEWS' and Model == 'ROBERTA':
        # for baseline training futher train the notmal ag news
        # pure_origin_model = 'textattack/roberta-base-ag-news'
        # pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/AGNEWS/ROBERTA/Pooled_AGNEWS/OT_GL_DAGNEWS_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/checkpoint-epoch-3/' #'textattack/roberta-base-ag-news'
        # pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/AGNEWS/ROBERTA/ARR_OCT_2023_Baseline/OT_GL_DAGNEWS_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-3'
        # pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/AGNEWS/ROBERTA/Baseline/OT_GL_DAGNEWS_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/checkpoint-epoch-3'
        if args.method == 'B':
            pure_origin_model = 'textattack/roberta-base-ag-news'
            # else:
            #     pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        elif args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            else:
                pure_origin_model =  '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        else:
            pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        






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
    if Dataset == 'SST2' and Model == 'ROBERTA':
        # pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/ARR_OCT_2023_Baseline/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        # pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/ARR_OCT_2023_Rigth_env_old_roberta_classification_head/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model'
        # pure_origin_model = "textattack/roberta-base-SST-2"

        
        
        if args.method == 'B':
            pure_origin_model = "textattack/roberta-base-SST-2"
            # else:
            #     pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        elif args.method == 'Embedding':
            if args.method_type =='DSRM':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            elif args.method_type =='ASCC':
                pure_origin_model = "roberta-base" # model has to be trained from scratch
            else:
                pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        
        else:
            pure_origin_model = '/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/SST2/ROBERTA/NAACL_Baseline_2/OT_GL_DSST2_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_DistNone_Dist_Val0_FLB0_0_OAT_None_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FF/last_model/'
        
        
        
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger' or attack_method[AttackTrain] == 'HotFlip' :
            parallel = False
        else:
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
        # pure_origin_model = "bert-base-uncased"

    if Dataset == 'YELP' and Model == 'ROBERTA':
        if args.method == 'B':
            pure_origin_model = 'textattack/roberta-base-uncased-yelp-polarity'
        else:
            # pure_origin_model = "VictorSanh/roberta-base-finetuned-yelp-polarity"
            # if can't find it, train B first
            pure_origin_model = "/home/brianformento/security_project/defense/attention_align/ot_alignment_attention/adv/LOCAL_Test/GLUE/YELP/ROBERTA/Pooled_YELP/OT_GL_DYELP_AKTTextFooler_MR0_3_ATETextFooler_B1_0_AT0_0_OT0_0_MMD0_0_CRL0_0_AUGNone_P0_0_PF0_0_FLB0_0_DR0_0_E3_LR2e-05_SS0_5_0_5_CS0_5_0_5_NC50_50_NWS0_FT/checkpoint-epoch-3"
        if attack_method[AttackTrain] == 'TextBugger' or attack_method[AttackEvaluate] == 'TextBugger':
            parallel = False
        else:
            parallel = True
        attack_num_workers_per_device = 8
        Epochs = [args.epochs]
        AEI = [args.epochs]
        Data_ratios = [args.data_ratio]
    if Dataset == 'MR' and Model == 'GPTJ':
        pure_origin_model = "hf-tiny-model-private/tiny-random-GPTJForCausalLM" # "EleutherAI/gpt-j-6B" #"EleutherAI/gpt-j-6B"
        # pure_origin_model = "ydshieh/tiny-random-gptj-for-sequence-classification" #"EleutherAI/gpt-j-6B"
        parallel = False
        attack_num_workers_per_device = 1
        Epochs = [args.epochs]
        AEI = [args.epochs]
        Data_ratios = [args.data_ratio]


    Frozen = args.frozen # 'True'

    Eval_method = args.eval_method #'epoch' #'last' # last, best

    dic_methods = { 'MMD':'MMD', 'CRL':'CRL', 'AT':'AT', 'OT':'OT','B':'B','AUG':'AUG','MMD_AT':'MMD_AT','MMD_P':'MMD_P','P':'P','PF':'PF','MMD_PF':'MMD_PF','FLB':'FLB','MMD_FLB':'MMD_FLB','Online_AT':'Online_AT','Dist':'Dist', 'Embedding':'Embedding'}
    method_test = dic_methods[args.method]


    # Data_ratios= [0.005,0.01,0.02,0.03,0.04,0.05] #[0.05,0.1,0.2,0.5,1.0]

    # Data_ratios = [16]



    if method_test == 'OT':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway
        # Lambdas = [{'B':1.0,'AT':0.0,'OT':0.001,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0}]

        # Lambdas = [{'B':1.0,'AT':0.0,'OT':1.0,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0},
        #            {'B':1.0,'AT':0.0,'OT':0.01,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0},
        #            {'B':1.0,'AT':0.0,'OT':0.1,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0}]

        # Lambdas = [{'B':1.0,'AT':0.0,'OT':1.0,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0}]

        if Dataset == 'MR':
            Lambdas = [{'B':1.0,'AT':0.0,'OT':0.001,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]

            # attention align uses 0.00001 = 1e-5 for roberta
            # Learning_rate = [0.01,0.1,0.001,0.00002] 
            # Learning_rate = [[0.00002],[0.001],[0.0001],[0.01]] #[0.01]
            # Learning_rate= [[0.00002]]
            Learning_rate= [[0.01]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.01,0.001]]    
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]         
        elif Dataset == 'IMDB':
            Learning_rate = [[0.01,0.001]]
        elif Dataset == 'AGNEWS':
            # Learning_rate = [[0.01]]
            Lambdas = [{'B':1.0,'AT':0.0,'OT':1.0,'MMD':0.,'CRL':0.,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]

            Learning_rate = [[0.00002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.01,0.001]]

    elif method_test == 'CRL':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

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
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.,'MMD':0.,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]

    elif method_test == 'AT':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':1.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'MMD':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'Dist':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':args.method_type,'Dist_Val':args.method_val,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'Embedding':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':0.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Embedding':args.method_type,'Embedding_Val':args.method_val,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'MMD_AT':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':1.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        # Lambdas = [{'B':0.1,'AT':5.0,'OT':0.0,'MMD':5.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0}]
        # Lambdas = [{'B':0.5,'AT':1.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.0,'PF':0.0,'FLB':0.0}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002]]
            # from freelb mrpc hyper param, 
            # Learning_rate = [[0.000005],[0.00001],[0.00003]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    
    elif method_test == 'AUG':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':'Adv','P':0.0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':None}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]


    elif method_test == 'MMD_P':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':10,'PF':None},
                    {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':5,'PF':None},
                    {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.5,'PF':None},
                    {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.1,'PF':None}]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'P':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':10,'PF':None},
                    {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':5,'PF':None},
                    {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.5,'PF':None},
                    {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.1,'PF':None}]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'PF':
        # Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':10,'PF':10},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':5,'PF':5},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.5,'PF':0.5},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0.1,'PF':0.1}]
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0,'PF':1},
                {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':1,'PF':1},
                {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':1,'PF':0}]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'MMD_PF':
        # Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':10,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':5,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.5,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.1,'PF':True}]

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0,'PF':1.0,'FLB':0.0} ]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'MMD_FLB':
        # Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':10,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':5,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.5,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.1,'PF':True}]

        Lambdas = [{'B':0.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'FLB':1.0} ]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'FLB':
        # Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':10,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':5,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.5,'PF':True},
        #             {'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0.1,'PF':True}]

        Lambdas = [{'B':0.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'Dist':None,'Dist_Val':0,'FLB':1.0,'FLB_Val':1.0,'Online_AT':None} ]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'AGNEWS':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'SST2':
            Learning_rate = [[0.00002,0.000002]]
    elif method_test == 'Online_AT':
        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':'TextFooler'},
                   {'B':1.0,'AT':1.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':'TextFooler'},
                    {'B':1.0,'AT':1.0,'OT':0.0,'MMD':1.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':'TextFooler'},
                     {'B':1.0,'AT':0.0,'OT':0.0,'MMD':0.0,'CRL':0.0,'AUG':None,'P':0,'PF':0.0,'FLB':0.0,'Dist':None,'Dist_Val':0,'Online_AT':'TextFooler'} ]
        # if P then we cycle on dataset and sample progressive samples, we then use
        # *P when using the distance metric (mmd, crl or OT)
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MRPC':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'MNLI': 
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'IMDB':
            Learning_rate = [[0.00002,0.000002]]
        elif Dataset == 'YELP':
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
                            origin_folder = pure_origin_model # "textattack/bert-base-uncased-rotten-tomatoes"




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
                                        
                                        # for New_Wildcard_AttackEvaluate in attack_method.keys():
                                        #     T_Name_New_Wildcard = f"{Id}_D{Dataset}_AKT{AttackTrain}_MR{max_modification_rate_adv}_ATE{New_Wildcard_AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_AUG{str(Lda['AUG'])}_P{Lda['P']}_PF{Lda['PF']}_FLB{Lda['FLB']}_DR{Data_ratio}_E{sum(Epochs[:ep+1])}_LR{LR[0]}_SS{sem_sim['train']}_{sem_sim['eval']}_CS{cos_sim['train']}_{cos_sim['eval']}_NC{no_cand['train']}_{no_cand['eval']}_NWS{NWS}_F{Frozen[0]}"
                                        #     T_Name_New_Wildcard = T_Name_New_Wildcard.replace('.','_')
                                        #     wild_card_path = archive_destination  + T_Name_Wildcard  
                                        clean_path = archive_destination  + T_Name
                                        # Train = False 
                                        print (f'Setting Train to False, Only evaluation is required')
                                        if os.path.isdir( clean_path ): 
                                            print ('Target Directory TK{AttackTrain}_ATE{AttackEvaluate} already exists, no need to copy, manualy delete this folder if checkpoints do not exist')
                                        else:
                                            Train = False
                                            # if Wildcard_AttackEvaluate == AttackEvaluate:# New_Wildcard_AttackEvaluate:
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
                                         
                                     
                                # check if output folder exists, in /nas/ if it dose skip trining and attack since we
                                # already did both.
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
                                    # changing the batch size
                                    if Data_ratios[k] < global_batch_size:
                                        batch_size = Data_ratios[k]
                                        warnings.warn(f'changing batch size to {batch_size} since number of provided samples is smaller than current batch size')
                                    else:
                                        batch_size = global_batch_size

                                # print ('data ratios',k,batch_size,Data_ratios[k],global_batch_size)


                                if Train:
                                    # get model_wrapper,development_attack,train_dataset,eval_dataset
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
                                                    num_clean_epochs=0, # used to be 1
                                                    num_train_adv_examples=Data_ratio,
                                                    learning_rate=Learning_rate[i][ep],
                                                    per_device_train_batch_size=batch_size,
                                                    gradient_accumulation_steps=accumulation_steps, # used to be 4
                                                    num_warmup_steps=NWS,
                                                    weight_decay=WD,
                                                    attack_epoch_interval=AEI[ep],
                                                    # log_to_txt=f"{result_path_symbol_final}",
                                                    log_to_csv=True,
                                                    log_to_tb = True,
                                                    csv_log_dir = f"{output_folder}/L.txt", # save it directly to /nas/
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
                                                    
                                                    query_budget_train = no_cand['train'], # this will be timed by the number of words in sentance (L) to give total quieries
                                                    
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


                                    # load new model
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
                                        # just load the last epoch when going into next learning rate
                                        origin_folder = output_folder + '/checkpoint-epoch-{Epochs[ep]}' # this dosn't have extension for next epoch load
                                        # we need to say either per epcoh, last or best
                                    else:
                                        origin_folder = output_folder # just set the origin folder to new test, we will load either per epoch,last or best
                                
                                


                                if Evaluate_Attack:


                                    if Train == False:
                                        origin_folder = output_folder

                                        parallel = False

                                            # initialize new model from memory
                                            # ID2 = Id +'_'+'D'+Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val +'_'+ 'MMD'+str_MMD_Val+ '_' + 'CRL'+str_CRL_Val + '_'+'DR'+str_Data_ratio + '_'+'E'+str_Epochs+ '_'+'LR'+str_LR+ '_'+'NWS'+str_NWS + '_'+'F'+str_F
                                            # ID = f"{Id}_D{Dataset}_AKT{AttackTrain}_ATE{AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_DR{Data_ratio}_E{Epochs}_LR{LR}_LR2{LR2}_NWS{NWS}_F{Frozen[0]}"
                                            # origin_folder = production_folder + T_Name
                                            # model_wrapper = get_model(origin_folder,Model,Frozen=Frozen)


                                    # if Train:
                                    #     model_wrapper = model_wrapper
                                    #
                                    #     # we have just trained so use same model
                                    # else:
                                    #     # initialize new model from memory
                                    #     # ID2 = Id +'_'+'D'+Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val +'_'+ 'MMD'+str_MMD_Val+ '_' + 'CRL'+str_CRL_Val + '_'+'DR'+str_Data_ratio + '_'+'E'+str_Epochs+ '_'+'LR'+str_LR+ '_'+'NWS'+str_NWS + '_'+'F'+str_F
                                    #     # ID = f"{Id}_D{Dataset}_AKT{AttackTrain}_ATE{AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_DR{Data_ratio}_E{Epochs}_LR{LR}_LR2{LR2}_NWS{NWS}_F{Frozen[0]}"
                                    #     origin_folder = production_folder + T_Name
                                    #     model_wrapper = get_model(origin_folder,Model,Frozen=Frozen)


                                    # dataset = get_eval_dataset(Dataset=Dataset,model_wrapper=model_wrapper, max_length= model_max_length )

                                    
                                    

                                    if Eval_method == 'epoch':
                                        # send number of epochs to attacker class,
                                        # have a attacker.attack_dataset_loop within attacker class
                                        # pass the high level components to create file name in attacker_class
                                        # have a for loop epoch in attack_dataset_loop that generates the file name and calls attack_dataset_parallel

                                        # method, name_file = get_evaluation_freq(Eval_method,e)
                                        # attack_folder = os.path.join(origin_folder,method)
                                        # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                        # model_wrapper.model.eval()
                                        # model_wrapper.model.to(textattack.shared.utils.device)
                                        # evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate)
                                        #
                                        #
                                        # evaluation_attack.goal_function.model

                                        #
                                        # method, name_file = get_evaluation_freq(Eval_method,1)
                                        # attack_folder = os.path.join(origin_folder,method)
                                        # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                        # model_wrapper.model.eval()
                                        # model_wrapper.model.to(textattack.shared.utils.device)
                                        # evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate)
                                        # for e in range(2,Epochs[ep]+1):
                                        #     # method, name_file = get_evaluation_freq(Eval_method,e)
                                        #     # attack_folder = os.path.join(origin_folder,method)
                                        #     # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                        #     # model_wrapper.model.eval()
                                        #     # model_wrapper.model.to(textattack.shared.utils.device)
                                        #     #
                                        #     # evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate)
                                        #     r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
                                        #     # method, name_file = get_evaluation_freq(Eval_method,e)
                                        #     # attack_folder = os.path.join(origin_folder,method)
                                        #     # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                        #     method, name_file = get_evaluation_freq(Eval_method,e)
                                        #     attack_folder = os.path.join(origin_folder,method)
                                        #     model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                        #     model_wrapper.model.eval()
                                        #     model_wrapper.model.to(textattack.shared.utils.device)
                                        #     evaluation_attack.goal_function.model=model_wrapper

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
                                            textattack.shared.utils.set_seed(765) #(self.attack_args.random_seed) 
                                            indices = np.random.choice(len(dataset), 500, replace=False)
                                            # dataset = np.random.choice(dataset, size=(500,2), replace=False)
                                            
                                            # dataset = [x for i,x in enumerate(dataset) if i in set(indices)  ] 
                                            dataset.filter_by_indices(indices)

                                            r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
                                             
                                            # method, name_file = get_evaluation_freq(Eval_method,e)
                                            # attack_folder = os.path.join(origin_folder,method)
                                            # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                            # method, name_file = get_evaluation_freq(Eval_method,e)
                                            # attack_folder = os.path.join(origin_folder,method)
                                            # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                            # model_wrapper.model.eval()
                                            # model_wrapper.model.to(textattack.shared.utils.device)
                                            # evaluation_attack.goal_function.model=model_wrapper


                                            # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                            # model_wrapper.model.eval()
                                            # model_wrapper.model.to(textattack.shared.utils.device)
                                            # evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate)
                                            # r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
                                            #
                                            # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                            # model_wrapper.model.eval()
                                            # model_wrapper.model.to(textattack.shared.utils.device)
                                            # evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate)
                                            # r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
                                            #
                                            # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                            # model_wrapper.model.eval()
                                            # model_wrapper.model.to(textattack.shared.utils.device)
                                            # evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate)
                                            # r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
                                            # sys.exit()

                                            # model_wrapper = get_model(attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length)
                                            # model_wrapper.model.eval()
                                            # model_wrapper.model.to(textattack.shared.utils.device)
                                            # evaluation_attack.goal_function.model = model_wrapper

                                            # model_wrapper.model.to('cpu')
                                            # torch.cuda.empty_cache()
                                    elif Eval_method == 'last' or Eval_method == 'best':
                                        method, name_file = get_evaluation_freq(Eval_method,0)

                                        attack_folder = os.path.join(origin_folder,method)
                                        
                                        model_wrapper = get_model(args,attack_folder,Model,Frozen=Frozen,model_max_length=model_max_length,method_test=args.method_type)
                                        model_wrapper.model.eval()
                                        model_wrapper.model.to(textattack.shared.utils.device)
                                        dataset = get_eval_dataset(Dataset=Dataset,model_wrapper=model_wrapper, max_length= model_max_length )

                                        evaluation_attack = get_attack(Attack=AttackEvaluate,model_wrapper=model_wrapper,cos_sim=cos_sim['eval'],sem_sim = sem_sim['eval'],no_cand=no_cand['eval'],max_modification_rate = max_modification_rate_atk)
                                        
                                        textattack.shared.utils.set_seed(765) #(self.attack_args.random_seed) 
                                        indices = np.random.choice(len(dataset), 500, replace=False)
                                        # dataset = np.random.choice(dataset, size=(500,2), replace=False)
                                        
                                        # dataset = [x for i,x in enumerate(dataset) if i in set(indices)  ] 
                                        dataset.filter_by_indices(indices)
                                        
                                        
                                        r=attack_model(origin_folder,name_file,evaluation_attack,dataset,query_budget=no_cand['eval'],parallel=parallel, attack_num_workers_per_device = attack_num_workers_per_device)
 

                                        # model_wrapper.model.to('cpu')
                                    # target = archive_destination + T_Name
                                    # os.makedirs(target, exist_ok=True)
                                    origin_folder = attack_folder

                                    model_wrapper.model.train()
                                    model_wrapper.model.to(textattack.shared.utils.device)


                            # folders_to_move = os.listdir(production_folder)
                            # for f in folders_to_move:
                            #     target_folder = archive_destination + f
                            #     current_folder = production_folder + f
                            #     # print (current_folder,target_folder)
                            #     shutil.move(current_folder , target_folder)
