import numpy as np
import csv
import torch
import argparse

parser = argparse.ArgumentParser(description='analysis')
parser.add_argument('--method', default='None', type=str, choices=['None','WCE','oversampling'], help='fairness method')
args = parser.parse_args()

log_pt = torch.load('./results/{}/mri_seed0_sub_seed7_epochs30_bs4_lr0.001_decay0.001_single_img_log.pt'.format(args.method))

tpr = 'tpr'
fpr = 'fpr'
for i in log_pt:
    if (tpr in i) or (fpr in i):
        print(i)
        print(np.array(log_pt[i][0]))
        print('\n')
    else:
        print(i)
        print(log_pt[i])
        print('\n')
