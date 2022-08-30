import torch
import numpy as np
from utils import check_log_dir, make_log_name, set_seed
from argument import get_args
import os
from importlib import import_module


args = get_args()

def main():
    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dir_name = args.date

    save_dir = os.path.join("./trained_models", dir_name)
    log_dir = os.path.join("./results", dir_name)
    check_log_dir(save_dir)
    check_log_dir(log_dir)
    
    if args.method == 'LBC':
        Modality = import_module(f'trainers.lbc_trainer')
    else:
        Modality = import_module(f'trainers.{args.data}_trainer')
    trainer = Modality.Trainer(args, log_dir, log_name, save_dir)
    
    if args.mode == "eval":
        model_dict = torch.load(args.modelpath)
        trainer.model.load_state_dict(model_dict)
        trainer.evaluate(trainer.model, trainer.test_loader, trainer.criterion, trainer.cuda)
        #_,  eval_acc, bal_acc, acc_gender, roc_scores, roc_scores_male, roc_scores_female = trainer.evaluate(trainer.model, trainer.test_loader, trainer.criterion, trainer.cuda)

        # auroc, fpr, tpr = roc_scores
        # auroc_male, fpr_male, tpr_male = roc_scores_male
        # auroc_female, fpr_female, tpr_female = roc_scores_female
        # male_acc, female_acc = acc_gender
        # male_tp, male_tn = male_acc
        # female_tp, female_tn = female_acc
        # if args.with_mci == False:
        #     print('Acc : {:.4f}, Auroc : {:.4f}, Auroc_male : {:.4f}, Auroc_female : {:.4f}'.format(eval_acc, auroc, auroc_male, auroc_female))
        # else:
        # eval_acc_ , _ = eval_acc
        # print('Acc : {:.4f}, Auroc : {:.4f}'.format(eval_acc_, auroc))
        print('Evaluation Finished!')

    else:
        trainer.train(args.epochs)


if __name__ == "__main__":
    main()
