import numpy as np
import torch
import csv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='method_anal')
# parser.add_argument('--method', type=str, choices=['None','WCE','oversampling','reweighting'], help='fairness method')
parser.add_argument('--model-seed', default=8, type=int, choices=[1,8])
parser.add_argument('--data-seed', default=8, type=int, choices=[1,8])
parser.add_argument('--epochs', default=30, type=int)
args = parser.parse_args()

fieldnames = ['lr','decay','bs','acc','acc_std','bal_acc','bal_acc_std']
file_data = []

model_seed = args.model_seed
data_seed = args.data_seed
num = model_seed * data_seed

val = 'best_val_acc'
val_bal = 'best_val_bal_acc'
train_loss = 'train_loss'
train_acc = 'train_acc'
val_loss = 'val_loss'
val_acc = 'val_acc'

val_l = np.empty((num))
val_bal_l = np.empty((num))

loss_tr_l = np.empty((num, args.epochs))
acc_tr_l = np.empty((num, args.epochs))
loss_val_l = np.empty((num, args.epochs))
acc_val_l = np.empty((num, args.epochs))

for lr in [0.001]:
    for decay in [0.01]:
        for batch_size in [4]:
            file_dict = {}
            best_val_acc = 0
            best_val_bal_acc = 0
            loss_tr = np.zeros((args.epochs))
            acc_tr = np.zeros((args.epochs))
            loss_val = np.zeros((args.epochs))
            acc_val = np.zeros((args.epochs))
            for seed_a in range(model_seed):
                for seed_b in range(data_seed):
                
                    log_pt = torch.load('./results/gender/mri_seed{}_sub_seed{}_epochs{}_bs{}_lr{}_decay{}_single_img_log.pt'.format(seed_a, seed_b, args.epochs, batch_size, lr, decay))
                    for i in log_pt:
                        if val == i:
                            best_val_acc += log_pt[i][0]
                            val_l[seed_b + data_seed*seed_a] = log_pt[i][0]
                        elif val_bal == i:
                            best_val_bal_acc += log_pt[i][0]
                            val_bal_l[seed_b + data_seed*seed_a] = log_pt[i][0]
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
            trainloss = loss_tr / num
            valloss = loss_val / num
            trainacc = acc_tr / num
            valacc = acc_val / num

            acc_std = np.sqrt(np.sum((val_l*100 - best_val_acc)**2) / num)
            bal_acc_std = np.sqrt(np.sum((val_bal_l*100 - best_val_bal_acc)**2) / num)
            trainloss_std = np.sqrt(np.sum((loss_tr_l - trainloss)**2, axis=0)/num)
            valloss_std = np.sqrt(np.sum((loss_val_l - valloss)**2, axis=0)/num)
            trainacc_std = np.sqrt(np.sum((acc_tr_l - trainacc)**2, axis=0)/num)
            valacc_std = np.sqrt(np.sum((acc_val_l - valacc)**2, axis=0)/num)

            file_dict['lr'] = lr
            file_dict['decay'] = decay
            file_dict['bs'] = batch_size
            file_dict['acc'] = round(best_val_acc, 3)
            file_dict['acc_std'] = round(acc_std, 3)
            file_dict['bal_acc'] = round(best_val_bal_acc, 3)
            file_dict['bal_acc_std'] = round(bal_acc_std, 3)
            file_data.append(file_dict)

            print("model seed {} / data seed {}".format(model_seed, data_seed))
            print("lr {} / decay {} / bs {}".format(lr,decay,batch_size))
            print("val_acc {:.3f} \nbal_acc {:.3f}".format(best_val_acc, best_val_bal_acc))

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
            plt.xlabel('epoch')
            plt.ylim(0,1)
            plt.legend(loc= (1.05,0.5))

            
            plt.savefig('./results/gender/graph.png', bbox_inches='tight', pad_inches=0.3)
#    file_data.append({})


f = open("gender.csv","w")
writer = csv.DictWriter(f, fieldnames = fieldnames)

writer.writeheader()
writer.writerows(file_data)

f.close()
