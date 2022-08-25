import numpy as np
import torch
import csv
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='method_anal')
parser.add_argument('--method', type=str, choices=['dummy','None','WCE','oversampling','LBC_eopp','LBC_eo'], help='fairness method')
parser.add_argument('--model-seed', default=8, type=int, choices=[1,8])
parser.add_argument('--data-seed', default=8, type=int, choices=[1,8])
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--iters', default=10, type=int)
args = parser.parse_args()

fieldnames = ['lr','decay','bs','acc','acc_std', 'bal_acc', 'bal_acc_std','m_acc', 'm_acc_std',
                'f_acc', 'f_acc_std', 'm_tn', 'm_tn_std','m_tp', 'm_tp_std', 'f_tn', 'f_tn_std', 'f_tp','f_tp_std','EO', 'EO_std', 'EOPP', 'EOPP_std']
file_data = []

epochs = args.epochs
model_seed = args.model_seed
data_seed = args.data_seed
num = model_seed * data_seed
iters = args.iters

val = 'best_val_acc'
val_bal = 'best_val_bal_acc'
val_male = 'best_val_acc_male'
val_female = 'best_val_acc_female'
m_tn_str = 'val_male_tn'
m_tp_str = 'val_male_tp'
f_tn_str = 'val_female_tn'
f_tp_str = 'val_female_tp'

train_loss = 'train_loss'
train_acc = 'train_acc'
val_loss = 'val_loss'
val_acc = 'val_acc'

val_l = np.empty((num))
val_bal_l = np.empty((num))
val_male_l = np.empty((num))
val_female_l = np.empty((num))
m_tn_l = np.empty((num))
m_tp_l = np.empty((num))
f_tn_l = np.empty((num))
f_tp_l = np.empty((num))
eo_l = np.empty((num))
eopp_l = np.empty((num))

loss_tr_l = np.empty((num, args.epochs*iters))
acc_tr_l = np.empty((num, args.epochs*iters))
loss_val_l = np.empty((num, args.epochs*iters))
acc_val_l = np.empty((num, args.epochs*iters))

for lr in [0.001]:
    for decay in [0.001]:
        for batch_size in [4]:
            file_dict = {}
            best_val_acc = 0
            best_val_bal_acc = 0
            best_val_acc_male = 0
            best_val_acc_female = 0
            m_tn = 0
            m_tp = 0
            f_tn = 0
            f_tp = 0
            loss_tr = np.zeros((args.epochs*iters))
            acc_tr = np.zeros((args.epochs*iters))
            loss_val = np.zeros((args.epochs*iters))
            acc_val = np.zeros((args.epochs*iters))
            for seed_a in range(model_seed):
                for seed_b in range(data_seed):
                
                    log_pt = torch.load('./results/{}/mri_seed{}_sub_seed{}_epochs{}_bs{}_lr{}_decay{}_single_img_log.pt'.format(args.method, seed_a, seed_b, epochs, batch_size, lr, decay))
                    for i in log_pt:
                        if val == i:
                            best_val_acc += log_pt[i][iters-1]
                            val_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif val_bal == i:
                            best_val_bal_acc += log_pt[i][iters-1]
                            val_bal_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif val_male == i:
                            best_val_acc_male += log_pt[i][iters-1]
                            val_male_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif val_female == i:
                            best_val_acc_female += log_pt[i][iters-1]
                            val_female_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif m_tn_str == i:
                            m_tn += log_pt[i][iters-1]
                            m_tn_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif m_tp_str == i:
                            m_tp += log_pt[i][iters-1]
                            m_tp_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif f_tn_str == i:
                            f_tn += log_pt[i][iters-1]
                            f_tn_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif f_tp_str == i:
                            f_tp += log_pt[i][iters-1]
                            f_tp_l[seed_b + data_seed*seed_a] = log_pt[i][iters-1]
                        elif train_loss == i:
                            loss_tr += log_pt[i]
                            loss_tr_l[seed_b + data_seed*seed_a] = log_pt[i]
                        elif val_loss == i:
                            loss_val += log_pt[i]
                            loss_val_l[seed_b + data_seed*seed_a] = log_pt[i]
                        elif train_acc == i:
                            acc_tr += log_pt[i]
                            acc_tr_l[seed_b + data_seed*seed_a] = log_pt[i]
                        elif val_acc == i:
                            acc_val += log_pt[i]
                            acc_val_l[seed_b + data_seed*seed_a] = log_pt[i]
            
            best_val_acc = 100*(best_val_acc / num)
            best_val_bal_acc = 100*(best_val_bal_acc / num)
            best_val_acc_male = 100*(best_val_acc_male / num)
            best_val_acc_female = 100*(best_val_acc_female / num)
            m_tn = 100*(m_tn / num)
            m_tp = 100*(m_tp / num)
            f_tn = 100*(f_tn / num)
            f_tp = 100*(f_tp / num)
            trainloss = loss_tr / num
            valloss = loss_val / num
            trainacc = acc_tr / num
            valacc = acc_val / num

            acc_std = np.sqrt(np.sum((val_l*100 - best_val_acc)**2) / num)
            bal_acc_std = np.sqrt(np.sum((val_bal_l*100 - best_val_bal_acc)**2) / num)
            male_acc_std = np.sqrt(np.sum((val_male_l*100 - best_val_acc_male)**2) / num)
            female_acc_std = np.sqrt(np.sum((val_female_l*100 - best_val_acc_female)**2) / num)
            m_tn_std = np.sqrt(np.sum((m_tn_l*100 - m_tn)**2) / num)
            m_tp_std = np.sqrt(np.sum((m_tp_l*100 - m_tp)**2) / num)
            f_tn_std = np.sqrt(np.sum((f_tn_l*100 - f_tn)**2) / num)
            f_tp_std = np.sqrt(np.sum((f_tp_l*100 - f_tp)**2) / num)
            trainloss_std = np.sqrt(np.sum((loss_tr_l - trainloss)**2, axis=0)/num)
            valloss_std = np.sqrt(np.sum((loss_val_l - valloss)**2, axis=0)/num)
            trainacc_std = np.sqrt(np.sum((acc_tr_l - trainacc)**2, axis=0)/num)
            valacc_std = np.sqrt(np.sum((acc_val_l - valacc)**2, axis=0)/num)

            eo_l = 100*(abs(m_tn_l-f_tn_l) + abs(m_tp_l-f_tp_l))
            eopp_l = 100*(abs(m_tp_l-f_tp_l))
            eo = np.sum(eo_l) / num
            eopp = np.sum(eopp_l) / num
            eo_std = np.sqrt(np.sum((eo_l - eo)**2)/num)
            eopp_std = np.sqrt(np.sum((eopp_l - eopp)**2)/num)

            file_dict['lr'] = lr
            file_dict['decay'] = decay
            file_dict['bs'] = batch_size
            file_dict['acc'] = round(best_val_acc, 3)
            file_dict['acc_std'] = round(acc_std, 3)
            file_dict['bal_acc'] = round(best_val_bal_acc, 3)
            file_dict['bal_acc_std'] = round(bal_acc_std, 3)
            file_dict['m_acc'] = round(best_val_acc_male, 3)
            file_dict['m_acc_std'] = round(male_acc_std, 3)
            file_dict['f_acc'] = round(best_val_acc_female, 3)
            file_dict['f_acc_std'] = round(female_acc_std, 3)
            file_dict['m_tn'] = round(m_tn, 3)
            file_dict['m_tn_std'] = round(m_tn_std, 3)
            file_dict['m_tp'] = round(m_tp, 3)
            file_dict['m_tp_std'] = round(m_tp_std, 3)
            file_dict['f_tn'] = round(f_tn, 3)
            file_dict['f_tn_std'] = round(f_tn_std, 3)
            file_dict['f_tp'] = round(f_tp, 3)
            file_dict['f_tp_std'] = round(f_tp_std, 3)
            # file_dict['EO'] = round(100*(abs(m_tn-f_tn) + abs(m_tp-f_tp)), 3)
            # file_dict['EOPP'] = round(100*(abs(m_tp-f_tp)), 3)
            file_dict['EO'] = round(eo,3)
            file_dict['EO_std'] = round(eo_std, 3)
            file_dict['EOPP'] = round(eopp, 3)
            file_dict['EOPP_std'] = round(eopp_std, 3)
            file_data.append(file_dict)

            print("model seed {} / data seed {}".format(num, num))
            print("method = {} lr {} / decay {} / bs {}".format(args.method,lr,decay,batch_size))
            print("val_acc {:.3f} \nbal_acc {:.3f} \nmale_acc {:.3f} \nfemale_acc {:.3f}".format(best_val_acc, best_val_bal_acc, best_val_acc_male, best_val_acc_female))
            print("m_tn {:.3f} \nf_tn {:.3f} \nm_tp {:.3f} \nf_tp {:.3f}".format(m_tn,f_tn,m_tp,f_tp))
            print("EO {:.3f} \nEOPP {:3f}".format(eo, eopp))

            fig = plt.figure()

            x = np.arange(1, 31, 1)
            ax1 = plt.subplot(2,1,1)
            plt.errorbar(x, trainloss, yerr=trainloss_std, label='train')
            plt.errorbar(x, valloss, yerr=valloss_std, label='val')
            plt.xticks(visible=False)
            plt.ylabel('loss')
            plt.legend(loc= (1.05,0.5))

            ax2 = plt.subplot(2,1,2, sharex=ax1)
            plt.errorbar(x, trainacc, yerr=trainacc_std, label='train')
            plt.errorbar(x, valacc, yerr=valacc_std, label='val')
            plt.ylabel('accuracy')
            plt.xlabel('epoch&iter')
            plt.ylim(0,1)
            plt.legend(loc= (1.05,0.5))

            plt.suptitle('Method : {}'.format(args.method))
            plt.savefig('./results/{}/graph.png'.format(args.method), bbox_inches='tight', pad_inches=0.5)

f = open("{}.csv".format(args.method),"w")
writer = csv.DictWriter(f, fieldnames = fieldnames)

writer.writeheader()
writer.writerows(file_data)

f.close()
