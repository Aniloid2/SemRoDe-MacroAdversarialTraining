import sys
import numpy as np
AT = '0_0'
OT = '0_0'
MMD = '1_0'
CRL = '0_0'

if AT != '0_0' or OT != '0_0' or MMD != '0_0' or CRL != '0_0':
    # DR = '1_0'
    DR = '0_5'
    # DR = '0_05'
else:
    DR = '0_0'

if AT == '0_0' and OT == '0_0' and MMD == '0_0' and CRL == '0_0':
    Method = 'Baseline'
elif  AT == '1_0' and OT == '0_0':
    Method = 'Adversarial Training (AT) Baseline'
elif  AT == '0_0' and OT != '0_0':
    Method = 'Offline Optimal Transport'
elif  MMD == '1_0':
    Method = 'MMD Baseline'
elif  CRL == '1_0':
    Method = 'CORAL Baseline'

# NWS = [0,10,100]
EP = 50
NWS = 0
LR1 = 0.00002
LR2 = 0.000002
Frozen = 'T'

dataset = 'MR'
list_methods = []
dependency_name = '/nas/Optimal_transport'
# dependency_name = './'
# extra_dependency_name = '50_and_10_epochs' preliminary tests to check diff lrs
# extra_dependency_name = 'LR_OT_Test' stable test with different learning rates first 50 then diff lr next 10
# extra_dependency_name = 'Data_Hyper_Test' # OT hyper parameter test to show that ot 0.01 is best with best learning rate
extra_dependency_name = '50_and_10_epochs/mmd_stable/'
# extra_dependency_name = ''
# extra_dependency_name= 'Off_LR_EP'
# extra_dependency_name = 'LR_0_00005'
# for d in dataset:
#     for e in EP:
#         for i in LR:
#             for n in NWS:
for p in range(1,51,1):

    str_LR1 = str(LR1).replace('.','_')
    str_LR2 = str(LR2).replace('.','_')
    str_NWS = str(NWS).replace('.','_')
    str_EP = str(EP)
    csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_NWS{str_NWS}_F{Frozen}"
    file_name = f'{dependency_name}/GLUE/{dataset}/BERT/{extra_dependency_name}/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_NWS{str_NWS}_F{Frozen}/E{p}.txt'
    file = open(file_name,'r')

    lst = file.readlines()
    # print ('ein',e,i,n)
    metrics = lst[-9:]
    # print (metrics)
    origin_acc = metrics[3].strip().split(': ')[1][:-1]
    adv_acc = metrics[4].strip().split(': ')[1][:-1]
    attack_success = metrics[5].strip().split(': ')[1][:-1]


    # print ('metho',origin_acc,adv_acc)

    # best epoch
    # csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}"
    # file_name = f'./GLUE/{d}/BERT/{extra_dependency_name}/OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}/train_log.txt'
    #
    # file = open(file_name,'r')
    #
    # lst = file.readlines()
    # f_n = len(lst)
    #
    # l_n = 0
    # list_of_epochs = []
    # while l_n < f_n:
    #     # print ('line_by_line',lst[l_n][:10])
    #
    #     if lst[l_n][:10] == 'Best score':
    #         if AT == '0_0' and OT == '0_0':
    #             line_with_epoch = lst[l_n - 4]
    #             # print ('current epoch',line_with_epoch)
    #             list_of_epochs.append(line_with_epoch.strip())
    #         else:
    #             line_with_epoch = lst[l_n - 3]
    #             # print ('current epoch',line_with_epoch)
    #             list_of_epochs.append(line_with_epoch.strip())
    #     l_n+=1
    # # print ('fn ln',f_n,l_n)
    # # sys.exit()
    # # print ('ist of epochs',list_of_epochs)
    # useful_epoch = list_of_epochs[-1]

    # tpl = (origin_acc,adv_acc,useful_epoch,csv_name)

    tpl = (origin_acc,adv_acc,attack_success,p,csv_name)

    list_methods.append(tpl)


csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_NWS{str_NWS}_F{Frozen}"
file_name = f'{dependency_name}/GLUE/{dataset}/BERT/{extra_dependency_name}/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_NWS{str_NWS}_F{Frozen}/L.txt'
file = open(file_name,'r')

lst = file.readlines()

TL = []
Iterations = []
dead_iter = []

for iter,loss_l in enumerate(lst):

    total_loss = loss_l.split(',')[0]
    total_loss = float(total_loss.split(':')[1])
    learning_rate = loss_l.split(',')[2]
    learning_rate = float(learning_rate.split(':')[1])

    if iter == 0:
        first_LR = learning_rate
    if learning_rate == first_LR:
        dead_iter.append(iter)
    # if iter == 10:
    #     sys.exit()
    # sys.exit()
print (dead_iter)

iter_count = 0
for iter,loss_l in enumerate(lst):
    total_loss = loss_l.split(',')[0]
    total_loss = float(total_loss.split(':')[1])
    learning_rate = loss_l.split(',')[2]
    learning_rate = float(learning_rate.split(':')[1])
    if iter >= max(dead_iter):
        Iterations.append(iter_count)
        iter_count+=1
        TL.append(total_loss)



file.close()


EP = 10
NWS = 0
# LR = 0.000002


for p in range(1,11,1):

    # str_LR = str(LR).replace('.','_')
    str_NWS = str(NWS).replace('.','_')
    str_EP = str(EP)
    try:
        csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}_F{Frozen}"
        file_name = f'{dependency_name}/GLUE/{dataset}/BERT/{extra_dependency_name}/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}_F{Frozen}/E{p}.txt'
        file = open(file_name,'r')
    except Exception as e:
        csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR2}_NWS{str_NWS}_F{Frozen}"
        file_name = f'{dependency_name}/GLUE/{dataset}/BERT/{extra_dependency_name}/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR2}_NWS{str_NWS}_F{Frozen}/E{p}.txt'
        file = open(file_name,'r')
        print ('old lr format')

    lst = file.readlines()
    # print ('ein',e,i,n)
    metrics = lst[-9:]
    # print (metrics)

    origin_acc = metrics[3].strip().split(': ')[1][:-1]
    adv_acc = metrics[4].strip().split(': ')[1][:-1]
    attack_success = metrics[5].strip().split(': ')[1][:-1]


    tpl = (origin_acc,adv_acc,attack_success,p+50,csv_name)

    list_methods.append(tpl)

# get also the training loss

sorted_tuples = sorted(list_methods ,  key=lambda x: x[3])
OA = []
AA = []
ASR = []
epochs = []
for i in sorted_tuples:
    OA.append(float(i[0]))
    AA.append(float(i[1]))
    ASR.append(float(i[2]))
    epochs.append(int(i[3]))
    print (i)



print ('mean OA',sum(OA)/len(OA),'mean AA',sum(AA)/len(AA))
print ('min OA',min(OA),'min AA',min(AA))
print ('max OA',max(OA),'max AA',max(AA))

try:
    csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}_F{Frozen}"
    file_name = f'{dependency_name}/GLUE/{dataset}/BERT/{extra_dependency_name}/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}_F{Frozen}/L.txt'
    file = open(file_name,'r')
except Exception as e:
    csv_name = f"OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR2}_NWS{str_NWS}_F{Frozen}"
    file_name = f'{dependency_name}/GLUE/{dataset}/BERT/{extra_dependency_name}/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR2}_NWS{str_NWS}_F{Frozen}/L.txt'
    file = open(file_name,'r')


lst = file.readlines()
max_prev = max(Iterations)
for iter,loss_l in enumerate(lst):
    total_loss = loss_l.split(',')[0]
    total_loss = float(total_loss.split(':')[1])

    Iterations.append(max_prev +  iter)
    TL.append(total_loss)

file.close()

import matplotlib.pyplot as plt

fig, ((ax1),(ax2),(ax3),(ax4)) = plt.subplots(4, 1,figsize=(7, 14))

ax1.plot(
    epochs,
    OA
)

ax1.set_title(f'{Method}\nTraining AT:{AT} OT:{OT}')


ax1.set_ylabel('Original Accuracy')


ax2.plot(
    epochs,
    AA
)


ax2.set_ylabel('After Attack Accuracy')


ax3.plot(
    epochs,
    ASR
)

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Attack Success Rate')

ax4.plot(
    Iterations,
    TL
)

ax4.set_xlabel('Iterations')
ax4.set_ylabel('Loss')

fig.savefig(f'./Pictures/Stable_Data_Hyper_Test/MMD_Test/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}_F{Frozen}.png')
fig.savefig(f'./Pictures/Stable_Data_Hyper_Test/MMD_Test/OT_GL_D{dataset}_B1_0_AT{AT}_OT{OT}_MMD{MMD}_CRL{CRL}_DR{DR}_E{str_EP}_LR{str_LR1}_{str_LR2}_NWS{str_NWS}_F{Frozen}.eps')


# plot iteration
