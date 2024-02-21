import sys
import numpy as np

if __name__ == '__main__':
    import transformers
    from textattack.models.wrappers import HuggingFaceModelWrapper
    import textattack
    from datasets import load_dataset
    import argparse
    import torch
    import os
    import shutil
    import pandas as pd
    import sys
    import warnings
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MR", type=str, help="The dataset to use")
    parser.add_argument("--model", default="BERT", type=str, help="The model to use")
    parser.add_argument("--method", default="OT", type=str, help="the method to use")
    parser.add_argument("--save_space", default="OT_Weeklong_Test", type=str, help="where to save the tests")
    args = parser.parse_args()


    Dataset = args.dataset #'MR' # args.dataset # high level in bash script we pass as arg
    Model = args.model # 'BERT'# args.model

    if Dataset == 'MR' and Model == 'BERT':
        pure_origin_model = "textattack/bert-base-uncased-rotten-tomatoes"
    if Dataset == 'MRPC' and Model == 'BERT':
        pure_origin_model = "textattack/bert-base-uncased-MRPC"


    archive_dir = args.save_space # "OT_Weeklong_Test"
    archive_destination = f"/nas/Optimal_transport/GLUE/{Dataset}/{Model}/{archive_dir}/"
    output_dir = f'./GLUE/{Dataset}/{Model}/'
    Id = 'OT_GL'


    AttackTrain = 'TextFooler'
    AttackEvaluate = 'TextFooler'

    Train = True
    Evaluate_Attack = True


    Checkpoint = 1
    batch_size = 64
    accumulation_steps = 1
    NWS = 0
    WD = 0
    Epochs = [50,10] # [50,10]
    AEI = [50,10] # [50,10]
    Frozen = 'True'

    Eval_method = 'epoch' # last, best


    dic_methods = { 'MMD':'MMD', 'CRL':'CRL', 'AT':'AT', 'OT':'OT'}
    method_test = dic_methods[args.method]


    row_list = []
    row_list_epoch_wise = []

    Data_ratios= [0.05,0.1,0.2,0.5,1.0]

    if method_test == 'OT':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.01,'MMD':0.,'CRL':0.},
                   {'B':1.0,'AT':0.0,'OT':0.1,'MMD':0.,'CRL':0.},
                   {'B':1.0,'AT':0.0,'OT':0.001,'MMD':0.,'CRL':0.},
                   {'B':1.0,'AT':0.0,'OT':1.0,'MMD':0.,'CRL':0.}]

        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [0.01,0.1,0.001,0.00002]
            Learning_rate_2 = [ 0.00002,0.00002,0.000002,0.000002]
        elif Dataset == 'MRPC':
            Learning_rate = [0.00002,0.001,0.01,0.1]
            Learning_rate_2 = [0.000002,0.000002, 0.00002,0.00002,0.000002]

    elif method_test == 'CRL':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.,'MMD':0.,'CRL':1.0}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [0.00002]
            Learning_rate_2 = [0.000002]
        elif Dataset == 'MRPC':
            Learning_rate = [0.00002]
            Learning_rate_2 = [0.000002]
    elif method_test == 'MMD':
        #0.5 unecessary, 0.001 too probs, attention align didnt use it
        # my OT method works best with 0.01 anyway

        Lambdas = [{'B':1.0,'AT':0.0,'OT':0.,'MMD':1.0,'CRL':0.}]
        if Dataset == 'MR':
            # attention align uses 0.00001 = 1e-5 for roberta
            Learning_rate = [0.00002]
            Learning_rate_2 = [0.000002]
        elif Dataset == 'MRPC':
            Learning_rate = [0.00002]
            Learning_rate_2 = [0.000002]


    for i,LR in enumerate(Learning_rate):
        LR2 = Learning_rate_2[i]
        for j,Lda in enumerate(Lambdas):
            for k,Data_ratio in enumerate(Data_ratios):

                origin_folder = pure_origin_model # "textattack/bert-base-uncased-rotten-tomatoes"

                epochs_list = [i+1 for i in range(sum(Epochs))]
                OA = []
                AA = []
                ASR = []

                Iterations = []
                TL = []

                failed = False
                for ep,epoch in enumerate(Epochs):


                    T_Name = f"{Id}_D{Dataset}_AKT{AttackTrain}_ATE{AttackEvaluate}_B{Lda['B']}_AT{Lda['AT']}_OT{Lda['OT']}_MMD{Lda['MMD']}_CRL{Lda['CRL']}_DR{Data_ratio}_E{sum(Epochs[:ep+1])}_LR{LR}_LRP{LR2}_NWS{NWS}_F{Frozen[0]}"
                    T_Name = T_Name.replace('.','_')

                    for s_e in range(1,epoch+1):
                        name_epoch = f'/E{s_e}.txt'
                        archive = archive_destination + T_Name + name_epoch
                        # print ('archive',archive)
                        try:
                            file = open(archive,'r')
                        except Exception as e:
                            # warnings.warn(f'Failed (could not find this file) test: {T_Name}')
                            # print (T_Name)
                            failed = True

                        lst = file.readlines()

                        metrics = lst[-9:]

                        # check if test was completed
                        check_is_right_string = None
                        try:
                            check_is_right_string = metrics[0].strip().split(': ')[0]
                        except Exception as e:
                            # warnings.warn(f'Failed (couldent even extract some strings) test: {T_Name}')
                            # print ('metrics:', metrics)
                            # print (T_Name)
                            failed = True

                        if check_is_right_string is not None and check_is_right_string == "Number of successful attacks":
                            origin_acc = metrics[3].strip().split(': ')[1][:-1]
                            OA.append(float(origin_acc))
                            adv_acc = metrics[4].strip().split(': ')[1][:-1]
                            AA.append(float(adv_acc))
                            att_succ_rate = metrics[5].strip().split(': ')[1][:-1]
                            ASR.append(float(att_succ_rate))
                            pert_words = metrics[6].strip().split(': ')[1][:-1]
                            # number of words per input 7
                            queries = metrics[8].strip().split(': ')[1][:-1]


                            # print ('orig acc',origin_acc, adv_acc, T_Name)
                        else:
                            # warnings.warn(f'Failed (do to incomplete attacks) test: {T_Name}')
                            # print (T_Name)
                            failed = True

                        if failed:
                            print (T_Name,name_epoch)
                            current_row = {'OT':Lda['OT'], 'Data_ratio':Data_ratio, 'LR1':LR, 'LR2':LR2,'E':sum(Epochs[:ep+1]),'SE':s_e,'Org Acc':'-', 'Aft Att Acc':'-', 'Att Suc Rate':'-', 'Perturbed Words':'-', 'Queries':'-','T_Name':T_Name }
                        else:
                            current_row = {'OT':Lda['OT'], 'Data_ratio':Data_ratio, 'LR1':LR, 'LR2':LR2,'E':sum(Epochs[:ep+1]),'SE':s_e,'Org Acc':origin_acc, 'Aft Att Acc':adv_acc, 'Att Suc Rate':att_succ_rate,'Perturbed Words':pert_words, 'Queries':queries,'T_Name':T_Name }


                        if s_e == ep:
                            row_list_epoch_wise.append(current_row)
                            row_list.append(current_row)
                        else:
                            row_list_epoch_wise.append(current_row)


                        # plot graph

                    if not failed:
                        loss_file_name = archive_destination + T_Name + '/L.txt'
                        loss_file = open(loss_file_name,'r')
                        loss_file_lines = loss_file.readlines()

                        max_prev = max(Iterations,default=0)
                        for iter,loss_l in enumerate(loss_file_lines):
                            total_loss = loss_l.split(',')[0]
                            total_loss = float(total_loss.split(':')[1])

                            Iterations.append(max_prev+iter)
                            TL.append(total_loss)

                        loss_file.close()

                if failed == True:
                    continue

                fig, ((ax1),(ax2),(ax3),(ax4)) = plt.subplots(4, 1,figsize=(7, 14))
                # print (epochs_list,OA)
                # print ('after attack',AA)
                # print ('attack success rate',ASR)
                # print ('iterations',Iterations)
                # print ('los',TL)
                # print (T_Name)
                ax1.plot(
                    epochs_list,
                    OA,
                    alpha = 0.2,
                    color = 'k',

                )

                ax1.set_title(f"OT {Lda['OT']}, LR1 {LR}, LR2 {LR2}, Data Ratio {Data_ratio*100}%")
                plt.suptitle(f"Robust performance over epochs",fontsize=21)



                ax1.set_ylabel('Original Accuracy [%]')

                ax1.set_ylim(0,100)
                ax1.set_xlim(1,sum(Epochs))
                # ax2.invert_yaxis()
                ax1.set_yticks([i*10 for i in range(1,11)])
                ax1.set_xticks([1]+[i*10 for i in range(1,sum(Epochs)//10 + 1)])
                z1 = np.polyfit(epochs_list, OA, 2)
                p1 = np.poly1d(z1)

                #add trendline to plot
                ax1.plot(epochs_list, p1(epochs_list),
                linestyle='dashed',
                color = 'r',
                )


                ax2.plot(
                    epochs_list,
                    AA,
                    alpha = 0.2,
                    color = 'k',
                )


                ax2.set_ylabel('After Attack Accuracy [%]')
                ax2.set_ylim(0,100)
                ax2.set_xlim(1,sum(Epochs))
                # ax2.invert_yaxis()
                ax2.set_yticks([i*10 for i in range(1,11)])
                ax2.set_xticks([1]+[i*10 for i in range(1,sum(Epochs)//10 + 1) ])
                z2 = np.polyfit(epochs_list, AA, 2)
                p2 = np.poly1d(z2)

                #add trendline to plot
                ax2.plot(epochs_list, p2(epochs_list),linestyle='dashed',
                color = 'r')


                ax3.plot(
                    epochs_list,
                    ASR,
                    alpha = 0.2,
                    color = 'k',
                )

                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Attack Success Rate [%]')
                ax3.set_ylim(0,100)
                ax3.set_xlim(0,sum(Epochs))
                # ax3.invert_yaxis()
                ax3.set_yticks([i*10 for i in range(1,11)])
                ax3.set_xticks([1]+[i*10 for i in range(1,sum(Epochs)//10 + 1)])

                z3 = np.polyfit(epochs_list, ASR, 2)
                p3 = np.poly1d(z3)

                #add trendline to plot
                ax3.plot(epochs_list, p3(epochs_list),linestyle='dashed',
                color = 'r')

                ax4.plot(
                    Iterations,
                    TL,

                    alpha = 0.2,
                    color = 'k',
                )


                ax4.set_xlabel('Iterations')
                ax4.set_ylabel('Loss')

                png_folder = f'{output_dir}/{archive_dir}/Pictures/Stable_Data_Hyper_Test/PNG/'
                eps_folder = f'{output_dir}/{archive_dir}/Pictures/Stable_Data_Hyper_Test/ESP/'

                os.makedirs(os.path.dirname(png_folder), exist_ok=True)
                os.makedirs(os.path.dirname(eps_folder), exist_ok=True)
                fig.savefig(f'{png_folder}/{T_Name}.png')
                fig.savefig(f'{eps_folder}/{T_Name}.eps')
                # sys.exit()




    grid_search_df = pd.DataFrame(row_list)
    grid_search_epoch_wise_df = pd.DataFrame(row_list_epoch_wise)

    output_folder = output_dir + archive_dir + f'/{Id}_D{Dataset}_AKT{AttackTrain}_ATE{AttackEvaluate}.csv'
    output_folder_epochwise = output_dir + archive_dir + f'/{Id}_D{Dataset}_AKT{AttackTrain}_ATE{AttackEvaluate}_EW.csv'

    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    os.makedirs(os.path.dirname(output_folder_epochwise), exist_ok=True)
    grid_search_df.to_csv(output_folder)
    grid_search_epoch_wise_df.to_csv(output_folder_epochwise)
    # grid_search_df = pd.DataFrame(columns=['OT', 'Data_ratio', 'LR1', 'LR2','E','Org Acc', 'Aft Att Acc', 'Att Suc Rate'])

                    # sys.exit()




                    # best epoch
                    # csv_name = f"OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}"
                    # file_name = f'./GLUE/{d}/BERT/{extra_dependency_name}/OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}/train_log.txt'
                    #
                    # file = open(file_name,'r')
                    #
                    # lst = file.readlines()
                    # f_n = len(lst)
                    #
                    # sys.exit()




# AT = '0_0'
# OT = '0_01'
#
# if AT != '0_0' or OT != '0_0':
#     DR = '0_05'
# else:
#     DR = '0_0'
#
# # NWS = [0,10,100]
# EP = [10,25,50,100]
# # EP = [100]
# NWS = [0]
# LR = [0.01,0.001,0.0001,0.00001,0.000001]
# dataset = ['MR']
# list_methods = []
# extra_dependency_name = ''
# # extra_dependency_name= 'Off_LR_EP'
# for d in dataset:
#     for e in EP:
#         for i in LR:
#             for n in NWS:
#                 str_LR = str(i).replace('.','_')
#                 str_NWS = str(n).replace('.','_')
#                 str_EP = str(e)
#                 csv_name = f"OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}"
#                 file_name = f'./GLUE/{d}/BERT/{extra_dependency_name}/OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}/B.txt'
#                 file = open(file_name,'r')
#
#                 lst = file.readlines()
#                 # print ('ein',e,i,n)
#                 metrics = lst[-9:]
#                 # print (metrics)
#                 origin_acc = metrics[3].strip().split(': ')[1][:-1]
#                 adv_acc = metrics[4].strip().split(': ')[1][:-1]
#
#
#                 # print ('metho',origin_acc,adv_acc)
#
#                 # best epoch
#                 csv_name = f"OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}"
#                 file_name = f'./GLUE/{d}/BERT/{extra_dependency_name}/OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}/train_log.txt'
#
#                 file = open(file_name,'r')
#
#                 lst = file.readlines()
#                 f_n = len(lst)
#
#                 l_n = 0
#                 list_of_epochs = []
#                 while l_n < f_n:
#                     # print ('line_by_line',lst[l_n][:10])
#
#                     if lst[l_n][:10] == 'Best score':
#                         if AT == '0_0' and OT == '0_0':
#                             line_with_epoch = lst[l_n - 4]
#                             # print ('current epoch',line_with_epoch)
#                             list_of_epochs.append(line_with_epoch.strip())
#                         else:
#                             line_with_epoch = lst[l_n - 3]
#                             # print ('current epoch',line_with_epoch)
#                             list_of_epochs.append(line_with_epoch.strip())
#                     l_n+=1
#                 # print ('fn ln',f_n,l_n)
#                 # sys.exit()
#                 # print ('ist of epochs',list_of_epochs)
#                 useful_epoch = list_of_epochs[-1]
#
#                 tpl = (origin_acc,adv_acc,useful_epoch,csv_name)
#
#                 list_methods.append(tpl)
#
# sorted_tuples = sorted(list_methods ,  key=lambda x: x[0])
# OA = []
# AA = []
# for i in sorted_tuples:
#     OA.append(float(i[0]))
#     AA.append(float(i[1]))
#     print (i)
# print ('mean OA',sum(OA)/len(OA),'mean AA',sum(AA)/len(AA))
# print ('min OA',min(OA),'min AA',min(AA))
# print ('max OA',max(OA),'max AA',max(AA))
