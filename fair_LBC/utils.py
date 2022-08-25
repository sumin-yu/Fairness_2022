import torch
import numpy as np
import random
import os
from os.path import join, expanduser
from scipy.io import loadmat
from typing import List
from sklearn import metrics
from sklearn.metrics import roc_curve
import pandas as pd

def list_dir(root: str, prefix: bool = False) -> List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_id_from_mri_folder_name(folder_name):
    start_idx = folder_name.find('-') + 1
    end_idx = folder_name.find('X')
    sub_id = folder_name[start_idx:end_idx]
    return sub_id


def get_mri_id(root):
    subject_data = list_dir(root)
    mri_sub = []
    for sub in subject_data:
        mri_sub.append(get_id_from_mri_folder_name(sub))
    return set(mri_sub)


def get_snp_id(snp_file):
    snp_data = pd.read_csv(snp_file)
    snp_sub = list(snp_data['IID'])
    snp_sub_refined = []
    for sub in snp_sub:
        sub = sub.replace('_','')
        snp_sub_refined.append(sub)
    return set(snp_sub_refined)


def get_accuracy(outputs, labels, binary=False, sigmoid_output=False, reduction='mean'):
    # if multi-label classification
    if len(labels.size()) > 1:
        outputs = (outputs > 0.0).float()
        correct = ((outputs == labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.detach()
    if binary :
        if sigmoid_output:
            predictions = (outputs >= 0.5).float()
        else:
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
    c = (predictions == labels).float().squeeze()
    if reduction == 'none':
        return c
    else:
        accuracy = torch.mean(c)
        return accuracy.detach()


def get_auroc(outputs, labels, binary=True):
    y_trues = torch.cat(labels).long()
    if binary :
        yhats = torch.cat(outputs).squeeze()
        y_preds = torch.sigmoid(yhats)
    else:
        yhats = torch.cat(outputs).squeeze()
        m = torch.nn.Softmax(dim=1)
        y_preds_tmp = m(yhats)
        y_preds = y_preds_tmp[:,1]
    roc_auc = metrics.roc_auc_score(y_true=y_trues, y_score=y_preds)
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    return roc_auc, fpr, tpr


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def make_log_name(args):
    log_name = ''

    if args.mode == 'eval':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]

    else:
        log_name += 'mri_'

        log_name += 'seed{}_sub_seed{}_epochs{}_bs{}_lr{}_decay{}'.format(args.seed, args.subject_seed, args.epochs,
                                                                          args.batch_size, args.lr, args.weight_decay)
        # if args.with_mci:
        #     log_name += '_gamma{}'.format(args.gamma)

        if args.use_single_img:
            log_name += '_single_img'

    return log_name


def save_model(state_dict, save_dir, log_name, is_best=False, fold=None):
    suffix = ''
    if fold is not None:
        suffix += '_fold{}'.format(fold)
    if is_best:
        suffix += '_best'
    suffix += '.pt'

    model_savepath = os.path.join(save_dir, log_name + suffix)
    torch.save(state_dict, model_savepath)
    print('Model saved to %s' % model_savepath)


def load_model(model, save_dir, log_name, load_best=False, fold=None):
    suffix = ''
    if fold is not None:
        suffix += '_fold{}'.format(fold)
    if load_best:
        suffix += '_best'
    suffix += '.pt'

    state_dict_path = os.path.join(save_dir, log_name + suffix)
    model.load_state_dict(torch.load(state_dict_path))
    return model


def get_joint_id():
    home = expanduser("~")
    mri_ids = get_mri_id(root=join(home, 'Data/BigBrain/ADNI/adni_registration_all/'))
    # mri id return ids with ad / cn
    file_name = 'ADNI_GWAS_522SNPs_QCed_03082022.csv'

    snp_ids = get_snp_id(snp_file=join(home, 'Data/BigBrain/ADNI/adni_snp/', file_name))
    joint_ids = set.intersection(mri_ids, snp_ids)
    return joint_ids
