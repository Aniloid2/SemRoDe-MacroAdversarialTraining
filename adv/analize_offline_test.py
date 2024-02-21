import sys
import numpy as np
AT = '0_0'
OT = '0_01'

if AT != '0_0' or OT != '0_0':
    DR = '0_05'
else:
    DR = '0_0'

# NWS = [0,10,100]
EP = [10,25,50,100]
# EP = [100]
NWS = [0]
LR = [0.01,0.001,0.0001,0.00001,0.000001]
dataset = ['MR']
list_methods = []
extra_dependency_name = ''
# extra_dependency_name= 'Off_LR_EP'
for d in dataset:
    for e in EP:
        for i in LR:
            for n in NWS:
                str_LR = str(i).replace('.','_')
                str_NWS = str(n).replace('.','_')
                str_EP = str(e)
                csv_name = f"OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}"
                file_name = f'./GLUE/{d}/BERT/{extra_dependency_name}/OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}/B.txt'
                file = open(file_name,'r')

                lst = file.readlines()
                # print ('ein',e,i,n)
                metrics = lst[-9:]
                # print (metrics)
                origin_acc = metrics[3].strip().split(': ')[1][:-1]
                adv_acc = metrics[4].strip().split(': ')[1][:-1]


                # print ('metho',origin_acc,adv_acc)

                # best epoch
                csv_name = f"OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}"
                file_name = f'./GLUE/{d}/BERT/{extra_dependency_name}/OT_GL_D{d}_B1_0_AT{AT}_OT{OT}_DR{DR}_E{str_EP}_LR{str_LR}_NWS{str_NWS}/train_log.txt'

                file = open(file_name,'r')

                lst = file.readlines()
                f_n = len(lst)

                l_n = 0
                list_of_epochs = []
                while l_n < f_n:
                    # print ('line_by_line',lst[l_n][:10])

                    if lst[l_n][:10] == 'Best score':
                        if AT == '0_0' and OT == '0_0':
                            line_with_epoch = lst[l_n - 4]
                            # print ('current epoch',line_with_epoch)
                            list_of_epochs.append(line_with_epoch.strip())
                        else:
                            line_with_epoch = lst[l_n - 3]
                            # print ('current epoch',line_with_epoch)
                            list_of_epochs.append(line_with_epoch.strip())
                    l_n+=1
                # print ('fn ln',f_n,l_n)
                # sys.exit()
                # print ('ist of epochs',list_of_epochs)
                useful_epoch = list_of_epochs[-1]

                tpl = (origin_acc,adv_acc,useful_epoch,csv_name)

                list_methods.append(tpl)

sorted_tuples = sorted(list_methods ,  key=lambda x: x[0])
OA = []
AA = []
for i in sorted_tuples:
    OA.append(float(i[0]))
    AA.append(float(i[1]))
    print (i)
print ('mean OA',sum(OA)/len(OA),'mean AA',sum(AA)/len(AA))
print ('min OA',min(OA),'min AA',min(AA))
print ('max OA',max(OA),'max AA',max(AA))

# for NSW and for LR
# do baseline
# need function that reads file and extracts original acc and after attack acc

#do at baseline

# do ot
