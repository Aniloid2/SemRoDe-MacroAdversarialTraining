if __name__ == '__main__':
    import transformers
    from textattack.models.wrappers import HuggingFaceModelWrapper
    import textattack
    from datasets import load_dataset
    import argparse
    # Original_40_run_prime
    # ID = 'ACT_40_run_prime'

    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    parser.add_argument("--model")
    parser.add_argument("--output_name")
    parser.add_argument("--B_Val",type=float)
    parser.add_argument("--AT_Val",type=float)
    parser.add_argument("--OT_Val",type=float)
    parser.add_argument("--MMD_Val",type=float)
    parser.add_argument("--CRL_Val",type=float)
    parser.add_argument("--Data_ratio",type=float)
    parser.add_argument("--Frozen")
    parser.add_argument("--Epochs",type=int)
    parser.add_argument("--E",type=int,help='current epoch to analyze')
    parser.add_argument("--Dataset",default='MRPC')
    parser.add_argument("--LR",type=float)
    parser.add_argument("--NWS",type=int)
    parser.add_argument("--Method" )
    args = parser.parse_args()

    if '/BERT/' in args.model or "bert-base-uncased" in args.model :
        Model_save = 'BERT'
    elif '/ALBERT/' in args.model or "albert-base-v2" in args.model :
        Model_save = 'ALBERT'

    if args.Method == 'last':
        method = 'last_model'
        name_file = 'R.txt'
    elif args.Method == 'best':
        method = 'best_model'
        name_file = 'B.txt'
    elif args.Method == 'epoch':
        method = f'checkpoint-epoch-{args.E}'
        name_file = f'E{args.E}.txt'


    str_B_Val = str(args.B_Val).replace('.','_')
    str_AT_Val = str(args.AT_Val).replace('.','_')
    str_OT_Val = str(args.OT_Val).replace('.','_')
    str_MMD_Val = str(args.MMD_Val).replace('.','_')
    str_CRL_Val = str(args.CRL_Val).replace('.','_')
    str_F = str(args.Frozen)[0]
    str_Data_ratio = str(args.Data_ratio).replace('.','_')
    str_Epochs = str(args.Epochs)
    str_LR = str(args.LR).replace('.','_')
    str_NWS = str(args.NWS).replace('.','_')

    ID = f'{args.id}'+'_'+'D'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val +'_'+ 'MMD'+str_MMD_Val+'_' +'CRL'+str_CRL_Val + '_'+'DR'+str_Data_ratio+'_'+'E'+str_Epochs +'_'+'LR'+str_LR+'_'+'NWS'+str_NWS +'_'+'F'+ str_F
    origin_folder = f"{args.model}{ID}/{method}/"
    output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID}"
    # output_name = ID


    # if 'OT_GL_CC' in args.id:
    #     str_B_Val = str(args.B_Val).replace('.','_')
    #     str_AT_Val = str(args.AT_Val).replace('.','_')
    #     str_OT_Val = str(args.OT_Val).replace('.','_')
    #     str_Data_ratio = str(args.Data_ratio).replace('.','_')
    #     str_Epochs = str(args.Epochs)
    #     ID = 'OT_GL_CC'+'_'+'DS'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val + '_'+'DR'+str_Data_ratio+'_'+'E'+str_Epochs
    #     origin_folder = f"{args.model}{ID}/{method}/"
    #     output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID}"
    #     output_name = ID
    # elif 'OT_GL' in args.id:
    #     str_B_Val = str(args.B_Val).replace('.','_')
    #     str_AT_Val = str(args.AT_Val).replace('.','_')
    #     str_OT_Val = str(args.OT_Val).replace('.','_')
    #     str_Data_ratio = str(args.Data_ratio).replace('.','_')
    #     str_Epochs = str(args.Epochs)
    #     ID = 'OT_GL'+'_'+'DS'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val + '_'+'DR'+str_Data_ratio+'_'+'E'+str_Epochs
    #     origin_folder = f"{args.model}{ID}/{method}/"
    #     output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID}"
    #     output_name = ID
    if 'Raw' in args.id:
        origin_folder = args.model

        ID = 'Raw'+'_'+'E'+str_Epochs
        output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID}"
        output_name = ID
    # else:
    #     ID = args.id
    #     origin_folder = args.model + '{ID}'
    #     output_folder = f"{args.model}{ID}"
    #     output_name = ID


    # origin_folder = f"../original_results/GLUE/MRPC/ALBERT/{ID}"
    # ID = 'Original_adv_learn'
    # origin_folder = f"./GLUE/MRPC/ALBERT/{ID}/{method}/"
    # https://huggingface.co/textattack
    model = transformers.AutoModelForSequenceClassification.from_pretrained(origin_folder)

    attack_bool = {"attack":True}

    model.config.update(attack_bool)
    print ('model configuration',model.config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(origin_folder,model_max_length=128)

    # We wrap the model so it can be used by textattack


    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MRPC")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MRPC")



    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


    # dataset_load = load_dataset("glue", "mrpc", split="test")
    # dataset = textattack.datasets.HuggingFaceDataset(dataset_load)

    if args.Dataset == 'MRPC':
        dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test")
    elif args.Dataset == 'MNLI':
        dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched",None,{0: 1, 1: 2, 2: 0},shuffle=True)
        # implement balaced dataset in testing
    elif args.Dataset == 'SNLI':
        # dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "train[0:10000]", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        # dataset.filter_by_labels_([0,1,2])
        eval_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        eval_dataset.filter_by_labels_([0,1,2])
    elif args.Dataset == 'MR':
        dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:249]+test[-250:]')

        # dataset.shuffle()
        c0 = 0
        c1 = 1
        for i,j in enumerate(dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('c0 , c1',c0,c1)

    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)


    # attack_args = textattack.AttackArgs(
    #     num_examples=500,
    #     log_to_txt=f"./{output_folder}/{name_file}",
    #     parallel=True,
    #     num_workers_per_device=8,
    #     disable_stdout=True,
    #     silent=True,
    # )
    attack_args = textattack.AttackArgs(
        num_examples=500,
        log_to_txt=f"./{output_folder}/{name_file}",
        parallel=False,
        num_workers_per_device=1,
        disable_stdout=True,
        silent=True,
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)



    attacker.attack_dataset()


    # print (model_wrapper)
    # print (dataset)
