# load 2 models
# -- 1st model is base model finetuned on just base dataset
# -- 2nd model is best model finetuned on OT 0.01,
# --- 20% adv data and 50 epochs on 0.01 LR and 10 epochs on 2*-5 LR

if __name__ == '__main__':
    import transformers
    from textattack.models.wrappers import HuggingFaceModelWrapper
    import textattack
    from datasets import load_dataset
    import argparse
    import collections
    import torch
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
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
    parser.add_argument("--Max_Words",type=int,default=None )
    parser.add_argument("--Min_Words",type=int,default=0 )
    parser.add_argument("--Boundary",type=float,default=0 )
    parser.add_argument('--Arrow', default=False, action='store_true')
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES=2 python analize_trajectory_tsn.py --model  "/nas/Optimal_transport/GLUE/MR/BERT/50_and_10_epochs/at_stable/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val  0.0 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.0 --Epochs 10 --E 10 --NWS 0 --LR 0.000002 --Frozen "True" --Method 'epoch' Boundary 0.6;

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
    train_batch_size = 1

    if str_Data_ratio == '0_0':
        M_Type = 'BaseT'
    else:
        M_Type = 'AdvT'


    textattack.shared.utils.set_seed(0)

    ID = f'{args.id}'+'_'+'D'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val +'_'+ 'MMD'+str_MMD_Val+'_' +'CRL'+str_CRL_Val + '_'+'DR'+str_Data_ratio+'_'+'E'+str_Epochs +'_'+'LR'+str_LR+'_'+'NWS'+str_NWS +'_'+'F'+ str_F
    origin_folder = f"{args.model}{ID}/{method}/"
    output_folder = f"./Pictures/{args.Dataset}/{Model_save}/{ID}"
    # output_name = ID


    # if 'OT_GL_CC' in args.id:
    #     str_B_Val = str(args.B_Val).replace('.','_')
    #     str_AT_Val = str(args.AT_Val).replace('.','_')
    #     str_OT_Val = str(args.OT_Val).replace('.','_')
    #     str_Data_ratio = str(args.Data_ratio).replace('.','_')
    #     str_Epochs = str(args.Epochs)
    #     ID = 'OT_GL_CC'+'_'+'DS'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val + '_'+'DR'+str_Data_ratio+'_'+'E'+str_Epochs
    #     origin_folder = f"{args.model}{ID}/{method}/"
    #     output_folder = f"./Pictures/{args.Dataset}/{Model_save}/{ID}"
    #     output_name = ID
    # elif 'OT_GL' in args.id:
    #     str_B_Val = str(args.B_Val).replace('.','_')
    #     str_AT_Val = str(args.AT_Val).replace('.','_')
    #     str_OT_Val = str(args.OT_Val).replace('.','_')
    #     str_Data_ratio = str(args.Data_ratio).replace('.','_')
    #     str_Epochs = str(args.Epochs)
    #     ID = 'OT_GL'+'_'+'DS'+args.Dataset+'_'+'B'+str_B_Val+'_'+'AT'+str_AT_Val+'_'+ 'OT'+str_OT_Val + '_'+'DR'+str_Data_ratio+'_'+'E'+str_Epochs
    #     origin_folder = f"{args.model}{ID}/{method}/"
    #     output_folder = f"./Pictures/{args.Dataset}/{Model_save}/{ID}"
    #     output_name = ID
    if 'Raw' in args.id:
        origin_folder = args.model

        ID = 'Raw'+'_'+'E'+str_Epochs
        output_folder = f"./Pictures/{args.Dataset}/{Model_save}/{ID}"
        output_name = ID
    # else:
    #     ID = args.id
    #     origin_folder = args.model + '{ID}'
    #     output_folder = f"{args.model}{ID}"
    #     output_name = ID


    # origin_folder = f"../original_results/Pictures/MRPC/ALBERT/{ID}"
    # ID = 'Original_adv_learn'
    # origin_folder = f"./Pictures/MRPC/ALBERT/{ID}/{method}/"
    # https://huggingface.co/textattack BertForSequenceClassificationOT
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(origin_folder)

    model = transformers.models.bert.BertForSequenceClassificationOT.from_pretrained(origin_folder)

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
        base_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:249]+test[-250:]')
        pre_adv_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:249]+test[-250:]')
        # base_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:19]+test[-20:]')
        # pre_adv_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:19]+test[-20:]')
        # base_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:499]+test[-500:]')
        # pre_adv_dataset = textattack.datasets.HuggingFaceDataset('rotten_tomatoes', None,'test[0:499]+test[-500:]')




        # plot base dataset class 0, class 1, plot adv dataset class 2, class 3
        # we are analysing robust model, we will have less samples that are adv
        # however, these perturbed samples may be closer to the base distribution than the adv samples from the non robust model
        # if this isn't the case, then we'll need to do a plot that highlights samples that havn't been missclassified anymore

        # dataset.shuffle()
        c0 = 0
        c1 = 1
        for i,j in enumerate(pre_adv_dataset):

            if j[-1] == 0:
                c0+=1
            elif j[-1] == 1:
                c1+=1
        print ('c0 , c1',c0,c1)


    # this for adv model
    # edit  the attack, so that we can pass a parameter to make it stronger or weaker
    # either by chaning the semantic sim val (threshold) or min_cos_sim
    # or maximum number of words

    # attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper,max_num_words=args.Max_Words, min_num_words=args.Min_Words)

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

    attacker = textattack.Attacker(attack, pre_adv_dataset, attack_args)

    from textattack.attack_results import MaximizedAttackResult, SuccessfulAttackResult,FailedAttackResult,SkippedAttackResult

    model.eval()
    my_file = Path(f"./cache_tsn_eval_samples_DS{args.Dataset}_DR{args.Data_ratio}_M{M_Type}_MinW{args.Min_Words}_MaxW{args.Max_Words}.pt")


    if my_file.is_file():
        adv_dataset = torch.load(f'./cache_tsn_eval_samples_DS{args.Dataset}_DR{args.Data_ratio}_M{M_Type}_MinW{args.Min_Words}_MaxW{args.Max_Words}.pt')
        # results.to(textattack.shared.utils.device)
        results = torch.load(f'./cache_tsn_results_eval_samples_DS{args.Dataset}_DR{args.Data_ratio}_M{M_Type}_MinW{args.Min_Words}_MaxW{args.Max_Words}.pt' )
        # pass results to logger
        from textattack.loggers import AttackLogManager
        Log = AttackLogManager()
        Log.log_results(results)

        for i,r in enumerate(results):
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult)):
                total_words = r.original_result.attacked_text.all_words_diff(
                        r.perturbed_result.attacked_text
                    )
                num_words_changed = len(total_words)
                print ('index',i,'num changed',num_words_changed)
                # if num_words_changed < 1 or num_words_changed >= 5:
                #     continue

                if i != 17:# 299:# 494:#299:
                    continue
                else:
                    number_of_mod = len(r.perturbed_result.attacked_text.attack_attrs['modified_indices'])
                    print ('indicies',r.perturbed_result.attacked_text.attack_attrs)
                    curr_end_sample = r.perturbed_result.attacked_text
                    extended_results = [(tuple(r.perturbed_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output)]
                    while curr_end_sample.attack_attrs.get('prev_attacked_text'):
                        next_sample = curr_end_sample.attack_attrs['prev_attacked_text']
                        extended_results.append((tuple(next_sample._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output ))
                        curr_end_sample = next_sample
                    # extended_results.append((tuple(r.original_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output ))
                    print ('extended results',extended_results)
                    extended_results.reverse()
                    break

        # print ('extended res',extended_results)
        # extended_results = [
        # (tuple(r.original_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output)
        # # (tuple( r.perturbed_result.attacked_text.attack_attrs['prev_attacked_text'].attack_attrs['prev_attacked_text'].attack_attrs['prev_attacked_text'].attack_attrs['prev_attacked_text']._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output )
        # # ,(tuple( r.perturbed_result.attacked_text.attack_attrs['prev_attacked_text']._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output )
        # ,(tuple(r.perturbed_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output)
        # ]
        # print ('last ext',extended_results)

        progress_dataset = textattack.datasets.Dataset(
            extended_results,
            input_columns=base_dataset.input_columns + ("_example_type",),
            label_map=base_dataset.label_map,
            label_names=base_dataset.label_names,
            output_scale_factor=base_dataset.output_scale_factor,
            shuffle=False,
        )
    else:
        results = attacker.attack_dataset()

        # print ('res1',results[0].original_result.attacked_text.attack_attrs)
        # print ('res1 text',results[0].original_result.attacked_text)
        # print ('res2',results[0].perturbed_result.attacked_text.attack_attrs)
        # print ('res2 text',results[0].perturbed_result.attacked_text.attack_attrs['prev_attacked_text'].attack_attrs['prev_attacked_text'].attack_attrs)
        # print (results[0].perturbed_result.attacked_text.attack_attrs['prev_attacked_text'].attack_attrs['prev_attacked_text'].attack_attrs.get('prev_attacked_text'))


        # number_of_mod = len(results[0].perturbed_result.attacked_text.attack_attrs['modified_indices'])
        # curr_end_sample = results[0].perturbed_result.attacked_text
        # extended_results = [results[0].perturbed_result.attacked_text]
        # while curr_end_sample.attack_attrs.get('prev_attacked_text'):
        #     next_sample = curr_end_sample.attack_attrs['prev_attacked_text']
        #     extended_results.append(next_sample)
        #     curr_end_sample = next_sample
        # extended_results.append(results[0].original_result.attacked_text)
        # print ('extended results',extended_results)
        # sys.exit()




        # print ('results',results)
        # from textattack.attack_results import MaximizedAttackResult, SuccessfulAttackResult

        # adversarial_examples = [
        #     (
        #         tuple(r.perturbed_result.attacked_text._text_input.values())
        #         + ("adversarial_example",),
        #         r.perturbed_result.ground_truth_output,
        #     )
        #     for r in results
        #     if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        # ]
        # print ('res',results)
        # print ('results',results)

        if args.Max_Words is not None:
            qualify_results = []
            for r in results:
                # print ('original  text:',r.original_result.attacked_text)
                # print ('perturbed text:',r.perturbed_result.attacked_text)

                total_words = r.original_result.attacked_text.all_words_diff(
                        r.perturbed_result.attacked_text
                    )
                # print ('total words',total_words)
                num_words_changed = len(total_words)
                # numwords > self.max_num_words) or numwords < self.min_num_words
                # print ('num words changed',num_words_changed,args.Min_Words,args.Max_Words)
                if num_words_changed < args.Min_Words or num_words_changed >= args.Max_Words:
                    continue
                else:
                    # print ('append')
                    qualify_results.append(r)
            results = qualify_results

        # we can plot the alpha by buckets
        # between 90% and 100% confidence is alpha 1
        # between 70-90 is alpha 0.7
        # between 50-70 alpha is 0.5


        adversarial_examples_failed = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (FailedAttackResult))
        ]

        adversarial_examples_success = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output, #+2,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ] # maybe add something to this tuple? maybe the entire connected componet
        # r.perturbed_result.attack_attrs

        # print ('success and fail',adversarial_examples_success,adversarial_examples_failed)

        # at this point we set labels of failed samples as 0,1 and of succ adv samles as 2,3
        # later we will add another 2 to the labels, so that the base samples have
        # labels 0,1, the failed adv samples label 2,3 and succ adv samples labels 4,5

        adversarial_examples = adversarial_examples_success #+ adversarial_examples_failed

        # define a small adv_dataset, where we only pass the final, progressive and original sample
        for i,r in enumerate(results):
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult)):
                total_words = r.original_result.attacked_text.all_words_diff(
                        r.perturbed_result.attacked_text
                    )
                num_words_changed = len(total_words)
                # print ('num changed',num_words_changed)
                # if num_words_changed < 3 or num_words_changed >= 4:
                #     continue
                if i != 7:#299:
                    continue
                else:
                    number_of_mod = len(r.perturbed_result.attacked_text.attack_attrs['modified_indices'])
                    print ('indicies',r.perturbed_result.attacked_text.attack_attrs)
                    curr_end_sample = r.perturbed_result.attacked_text
                    extended_results = [(tuple(r.perturbed_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output)]
                    while curr_end_sample.attack_attrs.get('prev_attacked_text'):
                        next_sample = curr_end_sample.attack_attrs['prev_attacked_text']
                        extended_results.append((tuple(next_sample._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output ))
                        curr_end_sample = next_sample
                    # extended_results.append((tuple(r.original_result.attacked_text._text_input.values())+ ("adversarial_example",),r.perturbed_result.ground_truth_output ))
                    # print ('extended results',extended_results)
                    extended_results.reverse()

                    progress_dataset = textattack.datasets.Dataset(
                        extended_results,
                        input_columns=base_dataset.input_columns + ("_example_type",),
                        label_map=base_dataset.label_map,
                        label_names=base_dataset.label_names,
                        output_scale_factor=base_dataset.output_scale_factor,
                        shuffle=False,
                    )
                    break



        # extended_results



        # print ('adv examples',adversarial_examples)
        # Name for column indicating if an example is adversarial is set as "_example_type".
        adv_dataset = textattack.datasets.Dataset(
            adversarial_examples,
            input_columns=base_dataset.input_columns + ("_example_type",),
            label_map=base_dataset.label_map,
            label_names=base_dataset.label_names,
            output_scale_factor=base_dataset.output_scale_factor,
            shuffle=False,
        ) # change the dataset so the adv data sample also has the data atts then have to change dataloader

        torch.save(adv_dataset,f'./cache_tsn_eval_samples_DS{args.Dataset}_DR{args.Data_ratio}_M{M_Type}_MinW{args.Min_Words}_MaxW{args.Max_Words}.pt' )
        torch.save(results,f'./cache_tsn_results_eval_samples_DS{args.Dataset}_DR{args.Data_ratio}_M{M_Type}_MinW{args.Min_Words}_MaxW{args.Max_Words}.pt' )

    # model.train()
    model.to(textattack.shared.utils.device)

    ###
    print (f'Max Words {args.Max_Words}, Min Words {args.Min_Words}, Number Adv Samples: {len(adv_dataset)}')







    # create data loader from these new adv samples

    # do the same for base dataset

    def get_train_dataloader( dataset, adv_dataset, batch_size):
        """Returns the :obj:`torch.utils.data.DataLoader` for training.

        Args:
            dataset (:class:`~textattack.datasets.Dataset`):
                Original training dataset.
            adv_dataset (:class:`~textattack.datasets.Dataset`):
                Adversarial examples generated from the original training dataset. :obj:`None` if no adversarial attack takes place.
            batch_size (:obj:`int`):
                Batch size for training.
        Returns:
            :obj:`torch.utils.data.DataLoader`
        """
        # TODO: Add pairing option where we can pair original examples with adversarial examples.
        # Helper functions for collating data
        def collate_fn(data):
            input_texts = []
            targets = []
            is_adv_sample = []

            for item in data:

                if "_example_type" in item[0].keys():

                    # Get example type value from OrderedDict and remove it

                    adv = item[0].pop("_example_type")

                    # with _example_type removed from item[0] OrderedDict
                    # all other keys should be part of input
                    _input, label = item
                    if adv != "adversarial_example":
                        raise ValueError(
                            "`item` has length of 3 but last element is not for marking if the item is an `adversarial example`."
                        )
                    else:
                        is_adv_sample.append(True)
                else:
                    # else `len(item)` is 2.
                    _input, label = item
                    is_adv_sample.append(False)

                if isinstance(_input, collections.OrderedDict):
                    _input = tuple(_input.values())
                else:
                    _input = tuple(_input)

                if len(_input) == 1:
                    _input = _input[0]
                input_texts.append(_input)
                targets.append(label)

            return input_texts, torch.tensor(targets), torch.tensor(is_adv_sample)

        if adv_dataset:
            dataset = torch.utils.data.ConcatDataset([dataset, adv_dataset])


        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,#True, #NEEDCHANGE# # need true or
            collate_fn=collate_fn,
            pin_memory=True,
            # drop_last =True,
        )
        return train_dataloader


    train_dataloader = get_train_dataloader(
        base_dataset,None,train_batch_size
    )

    base_prev_hidden = None
    base_prev_targets = None
    base_prev_pooler = None
    base_prev_confidence = None

    for step, base_batch in enumerate(train_dataloader):
        # input_texts, targets, is_adv_sample = batch
        base_input_texts, base_targets, base_is_base_sample = base_batch
        # print ('base in',base_input_texts)
        # sys.exit()
        base_input_ids = tokenizer(
            base_input_texts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        base_input_ids.to(textattack.shared.utils.device)

        with torch.no_grad():
            base_output = model(**base_input_ids ,return_dict=True,output_hidden_states=True) # need to extract last features
        # print  ('base out',base_output)
        base_logits = base_output[0]
        base_pooler_output = base_output.pooler_output # these are features from base batch need to append them somewhere
        # print ('pooler output shape',base_pooler_output)
        # print ('pool out',base_pooler_output)
        # print (base_output.hidden_states,len(base_output.hidden_states))
        # print ('12',base_output.hidden_states[12].shape)
        # print ('-1',base_output.hidden_states[-1].shape)#,base_output.hidden_states[11].shape,base_output.hidden_states[10].shape)
        base_last_hidden_states = base_output.hidden_states[-1]

        # print (last_hidden_states.shape)
        # print ('same 0',last_hidden_states[0,0])
        base_CLS_only = base_last_hidden_states[::,0,::]

        # print (CLS_only.shape)
        base_last_hidden_states = base_CLS_only


        # run classification to filter out missclassification and not plot them
        base_targets=base_targets.to(textattack.shared.utils.device)


        # save logits
        # print ('logit composition',base_logits)
        base_preds = base_logits.argmax(dim=-1)
        # print ('predicitons our of logits',base_preds)
        base_preds_sigmoid = torch.sigmoid(base_logits)
        # print ('preds sigmoid',base_preds_sigmoid)
        # print ('targets and preds',base_targets,base_preds)
        correct_predictions = (base_preds == base_targets)
        # print ('preds',correct_predictions)


        base_preds_sigmoid_max = base_preds_sigmoid.max(dim=-1).values


        # print ('max sigmoid',base_preds_sigmoid_max)





        # print ('len hidden states',len(base_last_hidden_states))
        # we only want to keep the samples which were correclty classified.
        base_last_hidden_states = base_last_hidden_states[correct_predictions]
        base_targets = base_targets[correct_predictions]
        base_pooler_output = base_pooler_output[correct_predictions]
        base_confidence_output = base_preds_sigmoid_max[correct_predictions]

        # len(base_batch)
        # base_indexes = [i+1+max(base_indexes) for i in range(len(base_batch))]
        # print ('bases', base_indexes[correct_predictions])
        # # sys.exit()
        # counter +=1
        # base_indexes.append(counter)
        # print (torch.Tensor([step]),correct_predictions,base_preds_sigmoid_max)

        if torch.is_tensor(base_prev_hidden):
            # print ('before 1',prev_hidden.shape,last_hidden_states.shape)
            base_prev_hidden = torch.cat((base_prev_hidden,base_last_hidden_states), dim=0)
            base_prev_targets = torch.cat((base_prev_targets,base_targets), dim=0)
            base_prev_pooler = torch.cat((base_prev_pooler,base_pooler_output), dim=0)
            base_prev_confidence = torch.cat((base_prev_confidence,base_confidence_output), dim=0)
            # print ('after 1',step,prev_hidden.shape )
        else:
            base_prev_hidden = base_last_hidden_states
            base_prev_targets = base_targets
            base_prev_pooler = base_pooler_output
            base_prev_confidence = base_preds_sigmoid_max
            # print ('after 0',step,prev_hidden.shape)

    # print ('confidence',base_prev_confidence)
    # base_prev_hidden = base_prev_hidden[:1]
    # base_prev_targets = base_prev_targets[:1]
    # base_prev_pooler = base_prev_pooler[:1]
    # base_prev_confidence = base_prev_confidence[:1]

    adv_dataloader = get_train_dataloader(
        adv_dataset,None,train_batch_size
    )

    prev_hidden = None
    prev_targets = None
    prev_pooler = None
    prev_confidence = None

    # in a loop cycle through the data loader
    # adv_cycle_dataloader = itertools.cycle(adv_dataloader)

    for step, adv_batch in enumerate(adv_dataloader):
        # input_texts, targets, is_adv_sample = batch
        adv_input_texts, adv_targets, adv_is_adv_sample = adv_batch

        adv_input_ids = tokenizer(
            adv_input_texts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        adv_input_ids.to(textattack.shared.utils.device)

        with torch.no_grad():
            adv_output = model(**adv_input_ids ,return_dict=True,output_hidden_states=True) # need to extract last features
        # print  ('adv out',adv_output)
        adv_logits = adv_output[0]
        adv_pooler_output = adv_output.pooler_output # these are features from adv batch need to append them somewhere



        # print ('pooler output shape',adv_pooler_output)
        # print ('pool out',adv_pooler_output)
        # print (adv_output.hidden_states,len(adv_output.hidden_states))
        # print ('12',adv_output.hidden_states[12].shape)
        # print ('-1',adv_output.hidden_states[-1].shape)#,adv_output.hidden_states[11].shape,adv_output.hidden_states[10].shape)
        last_hidden_states = adv_output.hidden_states[-1]

        # print (last_hidden_states.shape)
        # print ('same 0',last_hidden_states[0,0])
        CLS_only = last_hidden_states[::,0,::]

        # print (CLS_only.shape)
        last_hidden_states = CLS_only

        adv_targets=adv_targets.to(textattack.shared.utils.device)

        # print ('logit composition',adv_logits)
        adv_preds = adv_logits.argmax(dim=-1)
        adv_preds_sigmoid = torch.sigmoid(adv_logits)
        adv_preds_sigmoid_max = adv_preds_sigmoid.max(dim=-1).values
        # print ('adv preds sigmoid',adv_preds_sigmoid, base_preds_sigmoid_max)

        adv_successful = (adv_preds != adv_targets)




        last_hidden_states = last_hidden_states#[adv_successful]
        adv_targets = adv_targets#[adv_successful]

        adv_pooler_output = adv_pooler_output#[adv_successful]

        adv_preds_sigmoid_max = adv_preds_sigmoid_max#[adv_successful]



        if torch.is_tensor(prev_hidden):
            # print ('before 1',prev_hidden.shape,last_hidden_states.shape)
            prev_hidden = torch.cat((prev_hidden,last_hidden_states), dim=0)
            prev_targets = torch.cat((prev_targets,adv_targets), dim=0)
            prev_pooler = torch.cat((prev_pooler,adv_pooler_output), dim=0)
            prev_confidence = torch.cat((prev_confidence,adv_preds_sigmoid_max), dim=0)

        else:
            prev_hidden = last_hidden_states
            prev_targets = adv_targets
            prev_pooler = adv_pooler_output
            prev_confidence = adv_preds_sigmoid_max

    # prev_hidden = prev_hidden[:1]
    # prev_targets = prev_targets[:1]
    # prev_pooler = prev_pooler[:1]
    # prev_confidence = prev_confidence[:1]
    print ('adv prev confidence', prev_confidence)
    print ('adv',len(prev_pooler),len(prev_confidence))
    print ('base',len(base_prev_pooler),len(base_prev_confidence))


    progress_dataloader = get_train_dataloader(
        progress_dataset,None,train_batch_size
    )

    progress_prev_hidden = None
    progress_prev_targets = None
    progress_prev_pooler = None
    progress_prev_confidence = None
    # in a loop cycle through the data loader
    # adv_cycle_dataloader = itertools.cycle(adv_dataloader)

    for step, progress_batch in enumerate(progress_dataloader):
        # input_texts, targets, is_adv_sample = batch
        progress_input_texts, progress_targets, progress_is_adv_sample = progress_batch


        progress_input_ids = tokenizer(
            progress_input_texts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        progress_input_ids.to(textattack.shared.utils.device)

        with torch.no_grad():
            progress_output = model(**progress_input_ids ,return_dict=True,output_hidden_states=True) # need to extract last features

        progress_logits = progress_output[0]
        progress_pooler_output = progress_output.pooler_output



        progress_last_hidden_states = progress_output.hidden_states[-1]


        progress_CLS_only = progress_last_hidden_states[::,0,::]


        progress_last_hidden_states = progress_CLS_only

        progress_targets=progress_targets.to(textattack.shared.utils.device)


        progress_preds = progress_logits.argmax(dim=-1)
        progress_preds_sigmoid = torch.sigmoid(progress_logits)
        progress_preds_sigmoid_max = progress_preds_sigmoid.max(dim=-1).values

        progress_successful = (progress_preds != progress_targets)

        print ('preds',progress_preds)


        progress_last_hidden_states = progress_last_hidden_states#[progress_successful]
        progress_targets = progress_targets#[progress_successful]
        progress_pooler_output = progress_pooler_output#[progress_successful]

        progress_preds_sigmoid_max = progress_preds_sigmoid_max#[progress_successful]


        if torch.is_tensor(progress_prev_hidden):

            progress_prev_hidden = torch.cat((progress_prev_hidden,progress_last_hidden_states), dim=0)
            progress_prev_targets = torch.cat((progress_prev_targets,progress_targets), dim=0)
            progress_prev_pooler = torch.cat((progress_prev_pooler,progress_pooler_output), dim=0)
            progress_prev_confidence = torch.cat((progress_prev_confidence,progress_preds_sigmoid_max), dim=0)


        else:
            progress_prev_hidden = progress_last_hidden_states
            progress_prev_targets = progress_targets
            progress_prev_pooler = progress_pooler_output
            progress_prev_confidence = progress_preds_sigmoid_max


    print ('progress prev confidence', progress_prev_confidence)

    # # np.random.seed(0)
    # base_prev_hidden_np = base_prev_hidden.cpu().numpy()
    # base_prev_targets_np = base_prev_targets.cpu().numpy()
    # base_prev_pooler_np = base_prev_pooler.cpu().numpy()
    #
    # print ('prev target labels',base_prev_targets_np)
    # base_prev_targets_np = np.reshape(base_prev_targets_np,(len(base_prev_targets_np),1))
    # print ('new targets',base_prev_targets_np )
    # from sklearn.manifold import TSNE
    # base_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=0)
    # base_tsne_results = base_tsne.fit_transform(base_prev_hidden_np,base_prev_targets_np)
    # base_colors = base_prev_targets_np
    #
    #
    #
    #
    #
    #
    #
    # # numpy representation
    # prev_hidden_np = prev_hidden.cpu().numpy()
    # prev_targets_np = prev_targets.cpu().numpy()+2
    # prev_pooler_np = prev_pooler.cpu().numpy()
    # #
    # print ('prev target labels',prev_targets_np)
    # prev_targets_np = np.reshape(prev_targets_np,(len(prev_targets_np),1))
    # print ('new targets',prev_targets_np )
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=0)
    # tsne_results = tsne.fit_transform(prev_hidden_np,prev_targets_np)
    # colors = prev_targets_np
    # print (tsne_results)
    # print ('slices',tsne_results[::,0],tsne_results[::,1])
    #
    #
    # fig, ax = plt.subplots()
    # concatinated_tsn_x = np.concatenate((base_tsne_results[::,0],tsne_results[::,0]))
    # concatinated_tsn_y = np.concatenate((base_tsne_results[::,1],tsne_results[::,1]))
    #
    #
    #
    # concatinated_colors = np.concatenate((base_colors,colors))
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(concatinated_tsn_x,concatinated_tsn_y,c=concatinated_colors)
    #
    # # base_scatter = ax.scatter(base_tsne_results[::,0],base_tsne_results[::,1],c=base_colors)
    # # scatter = ax.scatter(tsne_results[::,0],tsne_results[::,1],c=colors)
    # # # ax.set_xlim([30, -30])
    # # # ax.set_ylim([30, -30])
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="lower left", title="Classes")
    # # ax.add_artist(legend1)
    # # legend2 = ax.legend(*base_scatter.legend_elements(),
    # #                 loc="lower left", title="Classes")
    # # ax.add_artist(legend2)
    # plt.savefig('./tsn_test_hidden_adv_and_base.png')


    # def



    # np.random.seed(0)
    base_prev_hidden_np = base_prev_hidden.cpu().numpy()
    base_prev_targets_np = base_prev_targets.cpu().numpy()
    base_prev_pooler_np = base_prev_pooler.cpu().numpy()
    base_prev_confidence_np = base_prev_confidence.cpu().numpy()

    # print ('prev target labels',base_prev_targets_np)
    base_prev_targets_np = np.reshape(base_prev_targets_np,(len(base_prev_targets_np),1))
    # print ('new targets',base_prev_targets_np )
    base_prev_confidence_np = np.reshape(base_prev_confidence_np,(len(base_prev_confidence_np),1))

    prev_hidden_np = prev_hidden.cpu().numpy()
    prev_targets_np = prev_targets.cpu().numpy()+2
    prev_pooler_np = prev_pooler.cpu().numpy()
    prev_confidence_np = prev_confidence.cpu().numpy()
    #
    # print ('prev target labels',prev_targets_np)

    # print ('base and prev',base_prev_targets_np,prev_targets_np)

    prev_targets_np = np.reshape(prev_targets_np,(len(prev_targets_np),1))
    # print ('new targets',prev_targets_np )


    prev_confidence_np = np.reshape(prev_confidence_np,(len(prev_confidence_np),1))
    # print ('prev conf', prev_confidence_np)

    concatinated_tsn_x = np.concatenate((base_prev_hidden_np,prev_hidden_np))
    # concatinated_tsn_y = np.concatenate((base_tsne_results[::,1],tsne_results[::,1]))

    concatinated_targets_np = np.concatenate((base_prev_targets_np,prev_targets_np))
    # print ('concatinated_targets', concatinated_targets)

    concatinated_confidence_np = np.concatenate((base_prev_confidence_np,prev_confidence_np))
    # print ('concatinate conf',concatinated_confidence)


    progress_prev_hidden_np = progress_prev_hidden.cpu().numpy()
    progress_prev_targets_np = progress_prev_targets.cpu().numpy() + 4
    progress_prev_confidence_np = progress_prev_confidence.cpu().numpy()
    progress_prev_targets_np = np.reshape(progress_prev_targets_np,(len(progress_prev_targets_np),1))
    progress_prev_confidence_np = np.reshape(progress_prev_confidence_np,(len(progress_prev_confidence_np),1))


    # if args.Arrow:
    concatinated_tsn_x = np.concatenate((concatinated_tsn_x,progress_prev_hidden_np))
    concatinated_targets_np = np.concatenate((concatinated_targets_np,progress_prev_targets_np))
    concatinated_confidence_np = np.concatenate((concatinated_confidence_np,progress_prev_confidence_np))
    new_base_colors = concatinated_targets_np
    new_alpha = concatinated_confidence_np



    from sklearn.manifold import TSNE
    new_base_tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000,random_state=0)
    new_base_tsne_results = new_base_tsne.fit_transform(concatinated_tsn_x,concatinated_targets_np)
    # new_base_colors = concatinated_targets_np
    # new_alpha = concatinated_confidence_np
    colors = []
    labels = []
    # fig, ax= plt.subplots()

    xs= new_base_tsne_results[::,0]
    ys = new_base_tsne_results[::,1]



    for e,i in enumerate(new_base_colors):
        if i == 0:
            labels.append('Base Negative Sample')
            colors.append('b')
            # scatter = ax.scatter(xs[i],ys[i] ,c='b',alpha=0.5,label='Base Negative Sample')
        if i == 1:
            labels.append('Base Positive Sample')
            colors.append('r')
            # scatter = ax.scatter(xs[i],ys[i] ,c='r',alpha=0.5,label='Base Positive Sample')
        if i == 2:
            labels.append('Adv Positive Sample (0)')
            colors.append('c')
            # scatter = ax.scatter(xs[i],ys[i] ,c='c',alpha=0.5,label='Adv Positive Sample (0->1)')
        if i == 3:
            labels.append('Adv Negative Sample (1)')
            colors.append('y')
            # scatter = ax.scatter(xs[i],ys[i] ,c='y',alpha=0.5,label='Adv Negative Sample (1->0)')

    # fig, ax = plt.subplots()
    # colors = ['b','r','c','y']
    # label=['Base Sample (0)','Base Sample (1)','Failed Adv Sample (Still 0)','Failed Adv Sample (Still 1)','Successful Adv Sample (0->1)','Successful Adv Sample (1->0)']

    label=['Base Sample (0)','Base Sample (1)','Successful Adv Sample (0->1)','Successful Adv Sample (1->0)','Boundary Sample']

    from matplotlib.colors import ListedColormap
    import seaborn as sns
    import random
    import pandas as pd


    x = new_base_tsne_results[::,0]
    y = new_base_tsne_results[::,1]
    df_new_alpha = np.squeeze(new_alpha)
    df_new_base_colors = np.squeeze(new_base_colors)

    len_progression = np.count_nonzero(df_new_base_colors == 4) +  np.count_nonzero(df_new_base_colors == 5)

    if args.Arrow:
        x = new_base_tsne_results[::,0]
        y = new_base_tsne_results[::,1]
        df_new_alpha = np.squeeze(new_alpha)
        df_new_base_colors = np.squeeze(new_base_colors)
    else:
        # print ('x y shape',x.shape,y.shape,len(df_new_base_colors),len(df_new_alpha))
        x = new_base_tsne_results[::,0][:-len_progression]
        y = new_base_tsne_results[::,1][:-len_progression]
        df_new_alpha = np.squeeze(new_alpha)[:-len_progression]
        df_new_base_colors = np.squeeze(new_base_colors)[:-len_progression]



    if args.Boundary:
        if args.Boundary < 1 and args.Boundary>0:

            df_new_base_colors = np.array([x if new_alpha[i] > args.Boundary else 6 for i,x in enumerate(df_new_base_colors)])
    else:
        df_new_base_colors = df_new_base_colors




    progression_x = x[-len_progression:]
    progression_y = y[-len_progression:]

    df = pd.DataFrame({'x': x, 'y': y, 'Sample Type':df_new_base_colors, 'Alpha':df_new_alpha})


    print ('dataframe',df.to_string())

    # scatter = ax.scatter(new_base_tsne_results[::,0],new_base_tsne_results[::,1],c=new_base_colors,alpha=0.9,cmap=cmaps)
    legend_map = {0:'Base Sample (0)',1:'Base Sample (1)',2:'Successful Adv Sample (0->1)',3:'Successful Adv Sample (1->0)',4:'Progression Sample (0->1)',5:'Progression Sample (1->0)'}
    # color_dictionary = {'Base Sample (0)':"#ADD8E6", "Base Sample (1)":"#FFFF00", 'Successful Adv Sample (0->1)':"#0000FF",'Successful Adv Sample (1->0)':"#FF0000",'Progression Sample (1->0)':"#000000",'Progression Sample (0->1)':"#000000"}
    # color_dictionary = {0:"#ADD8E6", 1:"#FFFF00", 2:"#0000FF",3:"#FF0000",5:"#000000",6:"#FFC0CB"}
    color_dictionary = {'Base Sample (0)':"#0000FF", # Blue
                        "Base Sample (1)":"#DC143C", #Crimson
                        'Successful Adv Sample (0->1)':"#87cefa", #lightskyblue
                        'Successful Adv Sample (1->0)':'#FFD700', #Gold
                        'Progression Sample (1->0)':"#000000",
                        'Progression Sample (0->1)':"#000000"}

    if args.Boundary:
        legend_map.update({6:'Boundary Sample'})
        color_dictionary.update({'Boundary Sample':"#808080"})


    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="x", y="y",alpha=df['Alpha'],hue=df['Sample Type'].map(legend_map),palette=color_dictionary,ax=ax1)
    # plt.legend(title='Sample Type', loc='upper left', labels=['Base Sample (0)','Base Sample (1)','Successful Adv Sample (0->1)','Successful Adv Sample (1->0)'])
    # ax1 = plot1.ax_joint()
    # fig1.annotate('20104 listings', xy=(progression_x[0], progression_y[0]),
    #         xytext=("Bronx", 16000),
    #         va='center',
    #         ha='center',
    #         arrowprops={'arrowstyle': '-|>', 'connectionstyle': 'arc3,rad=0.3'})

    if args.Arrow:
        for i in range(0,len_progression-1):
            ax1.annotate("", xy=(progression_x[i+1], progression_y[i+1]), xytext=(progression_x[i],progression_y[i]),
                arrowprops=dict(arrowstyle="->"))
    # ax.arrow(x=progression_x[1], y=progression_y[1], dx=progression_x[2]-progression_x[1], dy=progression_y[2]-progression_y[1], length_includes_head=True,head_width=1, head_length=1, width=0.3,color='black')




    # ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
    #         arrowprops=dict(arrowstyle="->"))

    # cp0 = [progression_x[0], progression_y[0]]
    # cp1 = [progression_x[1], progression_y[1]]
    # # unit vector in arrow direction
    # delta = cos, sin = (cp1 - cp0) / np.hypot(*(cp1 - cp0))
    # x_pos, y_pos = (
    #     (cp0 + cp1) / 2  # midpoint
    #     - delta * length / 2  # half the arrow length
    #     + np.array([-sin, cos]) * arrow_sep  # shift outwards by arrow_sep
    # )
    # ax.arrow(
    #     x_pos, y_pos, cos * length, sin * length,
    #     fc=fc, ec=ec or fc, alpha=alpha, width=width,
    #     head_width=head_width, head_length=head_length, shape=shape,
    #     length_includes_head=True,
    #     **kwargs
    # )


    # ax1.set_xlim([25, -25])
    # ax1.set_ylim([50, -25])


    # ax1.invert_xaxis()
    # ax1.invert_yaxis()



    if M_Type == 'AdvT':
        ax1.set_title(f'Robust T-SNE\nAdv Samples {len(adv_dataset)} MaxW{args.Max_Words} MinW{args.Min_Words}')
    elif M_Type == 'BaseT':
        ax1.set_title(f'Non-Robust T-SNE\nAdv Samples {len(adv_dataset)} MaxW{args.Max_Words} MinW{args.Min_Words}')
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="lower left", title="Classes")
    # ax.add_artist(legend1)
    # ax.legend()
    # legend1 = ax.legend(scatter.legend_elements(),
    #        labels=label,
    #        title="Distribution")
    # ax.add_artist(legend1)


    # plt.legend(handles=scatter.legend_elements()[0], labels=label )



    from pathlib import Path
    path_to_tsne = f"./Pictures/TSNE/{args.Dataset}/{Model_save}/"
    Path(path_to_tsne).mkdir(parents=True, exist_ok=True)
    fig1.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_Words}_MaxW{args.Max_Words}_Sing.png')
    fig1.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_Words}_MaxW{args.Max_Words}_Sing.eps')



    # import seaborn as sns
    # import random
    # import pandas as pd

    #
    # x = new_base_tsne_results[::,0]
    # y = new_base_tsne_results[::,1]


    # new_base_colors = np.squeeze(new_base_colors)



    # fig2, ax2= plt.subplots()
    df = pd.DataFrame({'x': x, 'y': y, 'Sample Type':df_new_base_colors})
    # color_dictionary = {'Base Sample (0)':"#ADD8E6", "Base Sample (1)":"#FFFF00", 'Successful Adv Sample (0->1)':"#0000FF",'Successful Adv Sample (1->0)':"#FF0000",'Progression Sample (1->0)':"#000000",'Progression Sample (0->1)':"#000000",'Boundary Sample':"#FFC0CB"}
    # we do the x and y limits due to a bug which leads to a distribution plotted also on the black points
    # plot2 = sns.jointplot(data = df, x = 'x', y = 'y',hue=df['Sample Type'].map(legend_map), kind = 'kde', xlim = (-25, 25), ylim = (-25, 50),palette=color_dictionary)
    plot2 = sns.jointplot(data = df, x = 'x', y = 'y',hue=df['Sample Type'].map(legend_map), kind = 'kde',palette=color_dictionary)
    # fig,ax = img.get_figure()
    # plt.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_words}_Joint.png')
    # plt.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_words}_Joint.eps')



    ax2 = plot2.ax_joint
    if args.Arrow:
        sns.scatterplot(data=df.iloc[-len_progression:], x="x", y="y",hue=df.iloc[-len_progression:]['Sample Type'].map(legend_map),palette=color_dictionary,ax=ax2,zorder =3)

        for i in range(0,len_progression-1):
            ax2.annotate("", xy=(progression_x[i+1], progression_y[i+1]), xytext=(progression_x[i],progression_y[i]),
                arrowprops=dict(arrowstyle="->"))



    # sns.scatterplot(data=df, x="x", y="y",hue="l",palette=["C0", "C1", "C2","C3"])
    plot2.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_Words}_MaxW{args.Max_Words}.png')
    plot2.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_Words}_MaxW{args.Max_Words}.eps')


    df = pd.DataFrame({'x': x, 'y': y, 'Sample Type':df_new_base_colors,'Alpha':df_new_alpha})

    # plot3 = sns.jointplot(data = df, x = 'x', y = 'y',hue=df['Sample Type'].map(legend_map), kind = 'kde', xlim = (-25, 25), ylim = (-25, 50),palette=color_dictionary)
    plot3 = sns.jointplot(data = df, x = 'x', y = 'y',hue=df['Sample Type'].map(legend_map), kind = 'kde', palette=color_dictionary)
    ax3 = plot3.ax_joint
    sns.scatterplot(data=df, x="x", y="y",alpha=df['Alpha'],hue=df['Sample Type'].map(legend_map),palette=color_dictionary,ax=ax3)
    if args.Arrow:
        for i in range(0,len_progression-1):
            ax3.annotate("", xy=(progression_x[i+1], progression_y[i+1]), xytext=(progression_x[i],progression_y[i]),
                arrowprops=dict(arrowstyle="->"))

    plot3.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_Words}_MaxW{args.Max_Words}_Joint.png')
    plot3.savefig(f'./Pictures/TSNE/{args.Dataset}/{Model_save}/{M_Type}_tsn_DS{args.Dataset}_DR{args.Data_ratio}_MinW{args.Min_Words}_MaxW{args.Max_Words}_Joint.eps')



# CUDA_VISIBLE_DEVICES=2 python analize_tsn.py --model  "/nas/Optimal_transport/GLUE/MR/BERT/Data_Hyper_Test/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val  0.01 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.2 --Epochs 10 --E 10 --NWS 0 --LR 0.00002 --Frozen "True" --Method 'epoch';
#
#
# CUDA_VISIBLE_DEVICES=2 python analize_tsn.py --model  "/nas/Optimal_transport/GLUE/MR/BERT/50_and_10_epochs/at_stable/" --id "OT_GL" --Dataset 'MR' --output_name "OT_GL" --B_Val 1 --AT_Val 0 --OT_Val  0.0 --MMD_Val 0  --CRL_Val 0 --Data_ratio 0.0 --Epochs 10 --E 10 --NWS 0 --LR 0.000002 --Frozen "True" --Method 'epoch';


# load base dataset# (can use the same code snipped from textattack_run_glue)
# take the evaluation part of this dataset
# perturb this evaluation dataset using both 1st and 2nd model using the attack and save it in cash as
# cache_evaluation_adv_samples_DSMR_base.pt,cache_evaluation_adv_samples_DSMR_robust.pt

# load the cashes, do feature extraction and do 2 plots.
# -- 1st plot  represents the base model, how well it clusters together the adv and base distributions
# -- 2nd plot represetns the adv model, it should cluster distributions better together.

#

# have an array that representas labels of class 1 from base, class 2 from base
# class 1 for adv and class 2 for adv

# plot 100 fetures from each
