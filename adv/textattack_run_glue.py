# def trainer(Model=None,
#             Id=None,
#             B_Val=None,
#             AT_Val=None,
#             OT_Val=None,
#             MMD_Val=None,
#             CRL_Val=None,
#             Frozen=None,
#             Data_ratio=None,
#             Epochs=None,
#             Dataset=None,
#             Attack=None,
#             BS=None,
#             AS=None,
#             NWS=None,
#             AEI=None,
#             LR=None,
#             WD=None,
#             Checkpoint=None
#
#         ):
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--model")
#     # parser.add_argument("--id")
#     # parser.add_argument("--B_Val")
#     # parser.add_argument("--AT_Val")
#     # parser.add_argument("--OT_Val")
#     # parser.add_argument("--MMD_Val")
#     # parser.add_argument("--CRL_Val")
#     # parser.add_argument("--Frozen")
#     # parser.add_argument("--Data_ratio")
#     # parser.add_argument("--Epochs")
#     # parser.add_argument("--Dataset")
#     # parser.add_argument("--Attack", help='Attack to use: TextFooler, PWWS')
#     # parser.add_argument("--BS",type=int,help='batch size to use')
#     # parser.add_argument("--AS",type=int,help='accumulation steps')
#     # parser.add_argument("--NWS",type=int, help='number warm up steps')
#     # parser.add_argument("--AEI",type=int, help='weight decay')
#     # parser.add_argument("--LR",type=float, help='learning rate')
#     # parser.add_argument("--WD",type=float, help='weight decay')
#     # parser.add_argument("--Checkpoint",type=int,default=None, help='every how many epochs to save model')
#     # parser.add_argument("--output_name")
#
#
#     # args = parser.parse_args()
#     # B_Val = float(args.B_Val)
#     # AT_Val = float(args.AT_Val)
#     # OT_Val = float(args.OT_Val)
#     # MMD_Val = float(args.MMD_Val)
#     # CRL_Val = float(args.CRL_Val)
#     # Data_ratio = float(args.Data_ratio)
#     # Epochs = int(args.Epochs)
#     #
#     # BS = args.BS
#     # AS = args.AS
#     # NWS = args.NWS
#     # LR = args.LR
#     # WD = args.WD
#     # AEI = args.AEI
#
#
#
#
#
#     str_B_Val = str(B_Val).replace('.','_')
#     str_AT_Val = str(AT_Val).replace('.','_')
#     str_OT_Val = str(OT_Val).replace('.','_')
#     str_MMD_Val = str(MMD_Val).replace('.','_')
#     str_CRL_Val = str(CRL_Val).replace('.','_')
#     str_F = str(args.Frozen)[0]
#
#     str_Data_ratio = str(Data_ratio).replace('.','_')
#     str_Epochs = str(Epochs)
#     str_LR = str(LR).replace('.','_')
#     str_NWS = str(NWS).replace('.','_')
#
#
#     if "bert-base-uncased" in args.model or "/BERT/" in args.model:
#         Model_save = 'BERT'
#     elif "albert-base-v2" in args.model:
#         Model_save = 'ALBERT'
#
#
#
#     ID2 = args.id +'_'+'D'+args.Dataset+'_'+'AKK'+args.Attack+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val +'_'+ 'MMD'+str_MMD_Val+ '_' + 'CRL'+str_CRL_Val + '_'+'DR'+str_Data_ratio + '_'+'E'+str_Epochs+ '_'+'LR'+str_LR+ '_'+'NWS'+str_NWS + '_'+'F'+str_F
#
#     output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID2}"
#     # origin_folder = f"textattack/bert-base-uncased-MRPC"
#
#
#     textattack.shared.utils.set_seed(786) # need to set seed outside to fix model training
#     # model = transformers.AutoModelForSequenceClassification.from_pretrained(origin_folder)
#     if Model_save == 'BERT':
#         model = transformers.models.bert.BertForSequenceClassificationOT.from_pretrained(args.model)
#         attack_bool = {"attack":True}
#         model.config.update(attack_bool)
#         print ('model configuration',model.config)
#     elif Model_save == 'ALBERT':
#         model = transformers.models.albert.AlbertForSequenceClassificationOT.from_pretrained(args.model)
#         attack_bool = {"attack":True}
#         model.config.update(attack_bool)
#         print ('how to freeze layers?')
#         sys.exit()
#
#     tokenizer = transformers.AutoTokenizer.from_pretrained(origin_folder,model_max_length=128)
#     model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
#
#     if args.Frozen == 'True':
#         print ('freezing', args.Frozen)
#         for i in range(0,11):
#             for name, param in model.named_parameters():
#                 if 'embeddings' in name:
#                     param.requires_grad = False
#                 check = f'.{i}.'
#                 if check in name:
#                     param.requires_grad = False
#     else:
#         print ('dont freeze', args.Frozen)
#         pass
#
#     if args.Dataset == 'MRPC':
#         train_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="train")
#         eval_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test")
#     elif args.Dataset == 'MNLI':
#         train_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',split="train[0:10000]",dataset_columns=None,label_map={0: 1, 1: 2, 2: 0},shuffle=True)
#
#         eval_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched",None,{0: 1, 1: 2, 2: 0},shuffle=True)
#     elif args.Dataset == 'SNLI':
#         train_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "train[0:10000]", None, {0: 1, 1: 2, 2: 0},shuffle=True)
#         train_dataset.filter_by_labels_([0,1,2])
#         eval_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
#         eval_dataset.filter_by_labels_([0,1,2])
#     elif args.Dataset == 'MR':
#         train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train')
#         # train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train[0:25]+train[-25:]')
#         train_dataset.shuffle()
#         eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test')
#
#
#     if args.Attack == 'TextFooler':
#         development_attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
#     elif args.Attack == 'PWWS':
#         development_attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
#
#
#
#     return output_folder, training_args
#
#
#
# if __name__ == '__main__':
#     trainer()
#




if __name__ == '__main__':
    import transformers
    from textattack.models.wrappers import HuggingFaceModelWrapper
    import textattack
    from datasets import load_dataset
    import argparse
    import torch
    import os



    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--model")
    parser.add_argument("--id")
    parser.add_argument("--B_Val")
    parser.add_argument("--AT_Val")
    parser.add_argument("--OT_Val")
    parser.add_argument("--MMD_Val")
    parser.add_argument("--CRL_Val")
    parser.add_argument("--Frozen")
    parser.add_argument("--Data_ratio")
    parser.add_argument("--Epochs")
    parser.add_argument("--Dataset")
    parser.add_argument("--Attack", help='Attack to use: TextFooler, PWWS')
    parser.add_argument("--BS",type=int,help='batch size to use')
    parser.add_argument("--AS",type=int,help='accumulation steps')
    parser.add_argument("--NWS",type=int, help='number warm up steps')
    parser.add_argument("--AEI",type=int, help='weight decay')
    parser.add_argument("--LR",type=float, help='learning rate')
    parser.add_argument("--WD",type=float, help='weight decay')
    parser.add_argument("--Checkpoint",type=int,default=None, help='every how many epochs to save model')
    # parser.add_argument("--output_name")
    args = parser.parse_args()
    B_Val = float(args.B_Val)
    AT_Val = float(args.AT_Val)
    OT_Val = float(args.OT_Val)
    MMD_Val = float(args.MMD_Val)
    CRL_Val = float(args.CRL_Val)
    Data_ratio = float(args.Data_ratio)
    Epochs = int(args.Epochs)

    BS = args.BS
    AS = args.AS
    NWS = args.NWS
    LR = args.LR
    WD = args.WD
    AEI = args.AEI


    if "bert-base-uncased" in args.model or "/BERT/" in args.model:
        Model_save = 'BERT'
    elif "albert-base-v2" in args.model:
        Model_save = 'ALBERT'

    # ID = 'Original_40_run_prime'
    # ID2 = args.id
    origin_folder = args.model

    str_B_Val = str(B_Val).replace('.','_')
    str_AT_Val = str(AT_Val).replace('.','_')
    str_OT_Val = str(OT_Val).replace('.','_')
    str_MMD_Val = str(MMD_Val).replace('.','_')
    str_CRL_Val = str(CRL_Val).replace('.','_')
    str_F = str(args.Frozen)[0]
    print (str_F)
    str_Data_ratio = str(Data_ratio).replace('.','_')
    str_Epochs =  str(Epochs)
    str_LR = str(LR).replace('.','_')
    str_NWS = str(NWS).replace('.','_')
    ID2 = args.id +'_'+'D'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val +'_'+ 'MMD'+str_MMD_Val+ '_' + 'CRL'+str_CRL_Val + '_'+'DR'+str_Data_ratio + '_'+'E'+str_Epochs+ '_'+'LR'+str_LR+ '_'+'NWS'+str_NWS + '_'+'F'+str_F


    output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID2}"
    # origin_folder = f"textattack/bert-base-uncased-MRPC"

    dump_file_name = f'{output_folder}/P.txt'
    os.makedirs(os.path.dirname(dump_file_name), exist_ok=True)
    file_object = open(dump_file_name,'a')
    parameter_dump = f"""Dataset:{args.Dataset}\nModel:{args.model}
    \nID type:{ID2}\nB_Val: {args.B_Val}\nAT_Val: {args.AT_Val}\nOT_Val:{args.OT_Val}\nMMD_Val:{args.MMD_Val}\nCRL_Val:{args.CRL_Val}
    \nData Ratio:{args.Data_ratio}\nEpochs:{args.Epochs}
    \n(BS) Batch Size:{args.BS}\n(AS) Accumulation steps:{args.AS}\n(WS) Warmup steps:{args.NWS}
    \n(LR) Learning Rate:{args.LR}\n(WD) Weight Decay:{args.WD}
    \n(Frozen) {args.Frozen}
    \n(AEI) Attack Epoch Interval:{args.AEI}
    \n(Attack) Attack type: args.Attack"""

    file_object.write(parameter_dump)
    file_object.close()

    # open text file and append all traiing args
    textattack.shared.utils.set_seed(786) ## need to set seed outside to fix model training
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(origin_folder)
    if Model_save == 'BERT':
        model = transformers.models.bert.BertForSequenceClassificationOT.from_pretrained(origin_folder)
        attack_bool = {"attack":True}
        model.config.update(attack_bool)
        print ('model configuration',model.config)
    elif Model_save == 'ALBERT':
        # config = transformers.AutoConfig.from_pretrained(
        # origin_folder,
        #     attack=True,
        #     inner_group_num=12,
        # )

        # model = transformers.models.albert.AlbertForSequenceClassificationOT.from_pretrained(origin_folder,config=config)
        model = transformers.models.albert.AlbertForSequenceClassificationOT.from_pretrained(origin_folder)
        # model_g = transformers.AlbertModel.from_pretrained(origin_folder)
        # model_f = transformers.AlbertF.from_pretrained(origin_folder)
        # attack_bool = {"attack":True,"inner_group_num":12}
        attack_bool = {"attack":True}
        model.config.update(attack_bool)
        print ('model configuration',model.config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(origin_folder,model_max_length=128)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # freeze up to the 11th layer

    if args.Frozen == 'True':
        print ('freezing', args.Frozen)
        for i in range(0,11):
            for name, param in model.named_parameters():
                if 'embeddings' in name:
                    param.requires_grad = False
                check = f'.{i}.'
                if check in name:
                    param.requires_grad = False
    else:
        print ('dont freeze', args.Frozen)
        pass



    #
    # for name, param in model.named_parameters():
    #         print ('name',name,param.requires_grad)
    # sys.exit()




    # train_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="train[3393:3395]")
    if args.Dataset == 'MRPC':
        train_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="train")
        eval_dataset = textattack.datasets.HuggingFaceDataset("glue", "mrpc", split="test")
    elif args.Dataset == 'MNLI':
        train_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',split="train[0:10000]",dataset_columns=None,label_map={0: 1, 1: 2, 2: 0},shuffle=True)

        eval_dataset = textattack.datasets.HuggingFaceDataset("glue",'mnli',"validation_matched",None,{0: 1, 1: 2, 2: 0},shuffle=True)
    elif args.Dataset == 'SNLI':
        train_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "train[0:10000]", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        train_dataset.filter_by_labels_([0,1,2])
        eval_dataset = textattack.datasets.HuggingFaceDataset( 'snli', None, "test", None, {0: 1, 1: 2, 2: 0},shuffle=True)
        eval_dataset.filter_by_labels_([0,1,2])
    elif args.Dataset == 'MR':
        train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train')
        # train_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'train[0:25]+train[-25:]')
        train_dataset.shuffle()
        eval_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test')

    # print (train_dataset)
    # for i,j in enumerate(train_dataset):
    #     print (i,j)
    # sys.exit()
    # TextFoolerJin2019
    # PWWSRen2019
    # if args.Attack == 'TextFooler':
    development_attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    # elif args.Attack == 'PWWS':
    #     development_attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)



    # if args.baseline == 'baseline':
    #     training_args = textattack.TrainingArgs(
    #                     num_epochs=4,
    #                     num_clean_epochs=4, # used to be 1
    #                     num_train_adv_examples=2,
    #                     learning_rate=2e-5,
    #                     per_device_train_batch_size=16,
    #                     gradient_accumulation_steps=1, # used to be 4
    #                     num_warmup_steps=5000,
    #                     weight_decay=0.01,
    #                     # log_to_txt=f"{result_path_symbol_final}",
    #                     output_dir=output_folder,
    #                 )
    # elif args.baseline == 'adv_training' :
    #     training_args = textattack.TrainingArgs(
    #                     num_epochs=4,
    #                     num_clean_epochs=0, # used to be 1
    #                     num_train_adv_examples=0.05,
    #                     learning_rate=2e-5,
    #                     per_device_train_batch_size=16,
    #                     gradient_accumulation_steps=1, # used to be 4
    #                     num_warmup_steps=5000,
    #                     weight_decay=0.01,
    #                     # log_to_txt=f"{result_path_symbol_final}",
    #                     output_dir=output_folder,
    #                 )
    # elif args.baseline == 'optimal_transport' :
    #     training_args = textattack.TrainingArgs(
    #                     num_epochs=4,
    #                     num_clean_epochs=0, # used to be 1
    #                     num_train_adv_examples=0.05,
    #                     learning_rate=2e-5,
    #                     per_device_train_batch_size=16,
    #                     gradient_accumulation_steps=1, # used to be 4
    #                     num_warmup_steps=5000,
    #                     weight_decay=0.01,
    #                     # log_to_txt=f"{result_path_symbol_final}",
    #                     output_dir=output_folder,
    #                 )
    if 'OT_GL_CC' == args.id: #args.baseline:
        # str_B_Val = str(B_Val).replace('.','_')
        # str_AT_Val = str(AT_Val).replace('.','_')
        # str_OT_Val = str(OT_Val).replace('.','_')
        # str_Data_ratio = str(Data_ratio).replace('.','_')
        # str_Epochs =  str(Epochs)
        # ID2 = 'OT_GL_CC'+'_'+'DS'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val + '_'+'DR'+str_Data_ratio + '_'+'E'+str_Epochs

        # output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID2}"
        if args.Dataset == 'MR':
            # batch_size = 64
            # accumulation_steps = 1
            batch_size = BS # 64
            accumulation_steps = AS # 1
        elif args.Dataset == 'MRPC':
            batch_size = BS # 64
            accumulation_steps = AS # 1


        training_args = textattack.TrainingArgs(
                        num_epochs=Epochs,
                        num_clean_epochs=0, # used to be 1
                        num_train_adv_examples=Data_ratio,
                        learning_rate=LR, #0.01
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=accumulation_steps, # used to be 4
                        num_warmup_steps=NWS,
                        weight_decay=WD, #0.001
                        attack_epoch_interval=AEI,
                        # log_to_txt=f"{result_path_symbol_final}",
                        log_to_csv=True,
                        log_to_tb = True,
                        csv_log_dir = f"{output_folder}/L.txt",
                        tb_log_dir = f"{output_folder}/runs/",
                        output_dir=output_folder,
                        B_Val = B_Val,
                        AT_Val = AT_Val,
                        OT_Val = OT_Val,
                        MMD_Val = MMD_Val,
                        CRL_Val = CRL_Val,
                        Data_ratio = Data_ratio,
                        Dataset_attack = args.Dataset,
                        checkpoint_interval_epochs = args.Checkpoint,
                    )

    elif 'OT_GL' == args.id: # args.baseline:
        # str_B_Val = str(B_Val).replace('.','_')
        # str_AT_Val = str(AT_Val).replace('.','_')
        # str_OT_Val = str(OT_Val).replace('.','_')
        # str_Data_ratio = str(Data_ratio).replace('.','_')
        # str_Epochs =  str(Epochs)
        if args.Dataset == 'MR':
            batch_size = args.BS # 64
            accumulation_steps = args.AS # 1
        elif args.Dataset == 'MRPC':
            batch_size = BS # 64
            accumulation_steps = AS # 1

        # ID2 = 'OT_GL'+'_'+'DS'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val + '_'+'DR'+str_Data_ratio + '_'+'E'+str_Epochs

        # output_folder = f"./GLUE/{args.Dataset}/{Model_save}/{ID2}"

        training_args = textattack.TrainingArgs(
                        num_epochs=Epochs,
                        num_clean_epochs=0, # used to be 1
                        num_train_adv_examples=Data_ratio,
                        learning_rate=LR,
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=accumulation_steps, # used to be 4
                        num_warmup_steps=NWS,
                        weight_decay=WD,
                        attack_epoch_interval=AEI,
                        # log_to_txt=f"{result_path_symbol_final}",
                        log_to_csv=True,
                        log_to_tb = True,
                        csv_log_dir = f"{output_folder}/L.txt",
                        tb_log_dir = f"{output_folder}/runs/",
                        output_dir=output_folder,
                        B_Val = B_Val,
                        AT_Val = AT_Val,
                        OT_Val = OT_Val,
                        MMD_Val=MMD_Val,
                        CRL_Val=CRL_Val,

                        Data_ratio = Data_ratio,
                        Dataset_attack = args.Dataset,
                        checkpoint_interval_epochs=args.Checkpoint,
                    )



    # if args.baseline == 'baseline' or args.baseline == 'adv_training':
    #     trainer = textattack.Trainer(
    #             model_wrapper,
    #             "classification",
    #             development_attack,
    #             train_dataset,
    #             eval_dataset,
    #             training_args
    #         )
    # elif  args.baseline == 'optimal_transport':
    #     trainer = textattack.Trainer(
    #             model_wrapper,
    #             "optimal_transport",
    #             development_attack,
    #             train_dataset,
    #             eval_dataset,
    #             training_args
    #         )
    if 'OT_GL_CC' == args.id:#  args.baseline:
        trainer = textattack.Trainer(
                model_wrapper,
                "OT_GL_CC",
                development_attack,
                train_dataset,
                eval_dataset,
                training_args
            )
    elif 'OT_GL' == args.id: # args.baseline:
        trainer = textattack.Trainer(
                model_wrapper,
                "OT_GL",
                development_attack,
                train_dataset,
                eval_dataset,
                training_args
            )


    trainer.train()
