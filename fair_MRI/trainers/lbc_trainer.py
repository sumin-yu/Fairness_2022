from __future__ import print_function

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from collections import defaultdict
import pickle
from torch.utils.data import DataLoader
from trainers.generic_trainer import GenericTrainer
from datasets.mri_dataset import MRIDataset
from networks.mobilenetv2 import MobileNetV2
from networks.shufflenetv2 import ShuffleNetV2
from networks.resnet import resnet18, resnet10
from os.path import join, expanduser
from torch.cuda.amp import GradScaler, autocast
from utils import get_accuracy, get_auroc, save_model, load_model

class Trainer(GenericTrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)

        self.eta = args.lbc_eta
        self.iteration = args.lbc_iteration
        self.batch_size = args.batch_size
        self.n_workers = args.num_workers
        self.reweighting_target_criterion = args.lbc_reweighting_target_criterion
        self.num_groups = 2
        self.num_classes = 3 if args.with_mci else 2

    def _get_dataset(self, args):
        mri_image_root = join(expanduser("~"), "./data/adni_registration_all/")
        train_dataset = MRIDataset(mri_image_root, split="train", seed=args.subject_seed, with_mci=args.with_mci,
                                   use_single_img=args.use_single_img, method=args.method
                                   )
        val_dataset = MRIDataset(mri_image_root, split="val", seed=args.subject_seed, with_mci=args.with_mci,
                                 use_single_img=args.use_single_img, method = args.method
                                 )
        test_dataset = MRIDataset(mri_image_root, split="test", seed=args.subject_seed, with_mci=args.with_mci,
                                  use_single_img=args.use_single_img, method=args.method
                                  )
        return train_dataset, val_dataset, test_dataset

    def _init_model(self, args):       
        num_outputs = 1 if not args.with_mci else 3
        if args.network == 'mobilenetv2':
            self.model = MobileNetV2(num_classes=num_outputs, sample_size=96, in_channel=1)
        elif args.network == 'shufflenetv2' :
            self.model = ShuffleNetV2(num_classes=num_outputs, sample_size=96, in_channel=1)
            # self.model = resnet10(pretrained=args.pretrained, num_classes=num_outputs, n_input_channels=1)

        if args.cuda:
            self.model = self.model.cuda()

    def train(self, epochs, model=None, train_loader=None, val_loader=None, test_loader=None, logger=None):
        model = self.model if model is None else model
        train_loader = self.train_loader if train_loader is None else train_loader
        val_loader = self.val_loader if val_loader is None else val_loader
        test_loader = self.test_loader if test_loader is None else test_loader

        optimizer, scheduler = self._get_optim_n_scheduler(self.args, model)
        scaler = GradScaler()
        logger = self.logger if logger is None else logger
        best_acc, best_bal_acc = 0.0, 0.0

        n_iters = self.iteration
        if self.num_classes > 2:
            self.extended_multipliers = torch.zeros((self.num_groups, self.num_classes))
        elif self.reweighting_target_criterion =='eopp' :
            self.extended_multipliers = torch.zeros((self.num_groups,))
        else: # eo
            self.extended_multipliers = torch.zeros((self.num_groups*2))

        # self.weight_matrix = self.get_weight_matrix(self.extended_multipliers, self.reweighting_target_criterion)
        self.weight_matrix = None

        print('epochs : ', epochs)
        print('eta_learning_rate : ', self.eta)
        print('n_iters : ', n_iters)
        
        violations = 0
        for iter_ in range(n_iters):
            start_t = time.time()

            for epoch in range(epochs):
                train_loss, train_acc, scaler = self._train_epoch(model, epoch, epochs, train_loader, optimizer, scaler)
                
                eval_start_time = time.time()
                val_loss, val_accs, val_roc_scores, val_f1, acc_male_v, acc_female_v, male_fair_v, female_fair_v = self.evaluate(
                    model, val_loader, self.criterion, self.cuda
                )
                val_acc, val_bal_acc = val_accs
                val_micro_F1, val_macro_F1 = val_f1
                val_auroc, val_fpr, val_tpr = val_roc_scores
                val_eo = max(abs(male_fair_v[0] - female_fair_v[0]), abs(male_fair_v[1] - female_fair_v[1]))
                val_eopp = abs(male_fair_v[1] - female_fair_v[1])

                test_loss, test_accs, test_roc_scores, test_f1, test_acc_male, test_acc_female, test_male_fair, test_female_fair = self.evaluate(
                    model, test_loader, self.criterion, self.cuda
                )

                test_acc, test_bal_acc = test_accs
                test_micro_F1, test_macro_F1 = test_f1
                test_auroc, test_fpr, test_tpr = test_roc_scores
                eval_end_time = time.time()

                print_state = '[{}/{}] Val Loss {:.3f} Test Loss: {:.3f}'.format(epoch + 1, epochs, val_loss, test_loss)
                print_state += ' Val Acc: {:.3f} Test Acc: {:.3f}'.format(val_acc, test_acc)
                
                if self.num_classes==3: # with-mci
                    print_state += ' Val F1 (micro/macro): {:.3f}/{:.3f} Test F1 (micro/macro): {:.3f}/{:.3f}'.format(
                        val_micro_F1, val_macro_F1, test_micro_F1, test_macro_F1)

                if self.binary:
                    print_state += ' Val AUROC: {:.3f} Test AUROC: {:.3f}'.format(val_auroc, test_auroc)
                print(print_state + ' [{:.3f} s]'.format(eval_end_time - eval_start_time))

                scheduler.step()
                self._update_log(
                    logger=logger, train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc, val_bal_acc=val_bal_acc,
                    val_acc_male=acc_male_v, val_acc_female=acc_female_v, val_male_tn=male_fair_v[0], val_male_tp=male_fair_v[1], val_female_tn=female_fair_v[0], val_female_tp=female_fair_v[1],
                    val_eo=val_eo, val_eopp=val_eopp,
                    test_loss=test_loss, test_acc=test_acc, test_bal_acc=test_bal_acc,
                )
                # eval_start_time = time.time()                
                # val_loss, val_accs, val_roc_scores, val_f1, val_acc_male, val_acc_female, val_male_fair, val_female_fair = self.evaluate(
                #     model, val_loader, self.criterion, self.cuda
                # )
                # val_acc, val_bal_acc = val_accs
                # val_micro_F1, val_macro_F1 = val_f1
                # val_auroc, val_fpr, val_tpr = val_roc_scores

                # test_loss, test_accs, test_roc_scores, test_f1, test_acc_male, test_acc_female, test_male_fair, test_female_fair = self.evaluate(
                #     model, test_loader, self.criterion, self.cuda
                # )
                # test_acc, test_bal_acc = test_accs
                # test_micro_F1, test_macro_F1 = test_f1
                # test_auroc, test_fpr, test_tpr = test_roc_scores
                            
                # eval_end_time = time.time()
                # print_state = '[{}/{}] Val Loss {:.3f} Test Loss: {:.3f}'.format(epoch + 1, epochs, val_loss, test_loss)
                # print_state += ' Val Acc: {:.3f} Test Acc: {:.3f}'.format(val_acc, test_acc)
                
                # if self.num_classes==3: # with-mci
                #     print_state += ' Val F1 (micro/macro): {:.3f}/{:.3f} Test F1 (micro/macro): {:.3f}/{:.3f}'.format(
                #         val_micro_F1, val_macro_F1, test_micro_F1, test_macro_F1)

                # if self.binary:
                #     print_state += ' Val AUROC: {:.3f} Test AUROC: {:.3f}'.format(val_auroc, test_auroc)
                # print(print_state + ' [{:.3f} s]'.format(eval_end_time - eval_start_time))

                # scheduler.step()  
                # self._update_log(
                #     logger=logger, train_loss=train_loss, train_acc=train_acc,
                #     val_loss=val_loss, val_acc=val_acc, val_bal_acc=val_bal_acc,
                #     test_loss=test_loss, test_acc=test_acc, test_bal_acc=test_bal_acc,
                # )

                if self.binary:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        save_model(model.state_dict(), self.save_dir, self.log_name, is_best=True)
                else:
                    if val_bal_acc > best_bal_acc:
                        best_bal_acc = val_bal_acc
                        save_model(model.state_dict(), self.save_dir, self.log_name, is_best=True)

            save_model(model.state_dict(), self.save_dir, self.log_name)

            # # get best model results
            # model = load_model(model, self.save_dir, self.log_name, load_best=True)

            # _, eval_accs, roc_scores, eval_f1, acc_male_v, acc_female_v, male_fair_v, female_fair_v = self.evaluate(model, val_loader, self.criterion, self.cuda)
            # val_acc, val_bal_acc = eval_accs
            # val_auroc, val_fpr, val_tpr = roc_scores
            # val_micro_F1, val_macro_F1 = eval_f1

            # _, eval_accs, roc_scores, eval_f1, acc_male_t, acc_female_t, male_fair_t, female_fair_t = self.evaluate(model, test_loader, self.criterion, self.cuda)
            # test_acc, test_bal_acc = eval_accs
            # test_auroc, test_fpr, test_tpr = roc_scores
            # test_micro_F1, test_macro_F1 = eval_f1

            # self._update_log(logger=logger, is_last=True,
            #                 best_val_acc=val_acc, best_val_bal_acc=val_bal_acc, best_val_acc_male=acc_male_v, best_val_acc_female=acc_female_v,
            #                 val_male_tn = male_fair_v[0], val_male_tp = male_fair_v[1], val_female_tn = female_fair_v[0], val_female_tp = female_fair_v[1],
            #                 best_val_auroc=val_auroc, best_val_fpr=val_fpr, best_val_tpr=val_tpr,
            #                 best_test_acc=test_acc, best_test_bal_acc=test_bal_acc, best_test_acc_male=acc_male_t, best_test_acc_female=acc_female_t,
            #                 test_male_tn = male_fair_t[0], test_male_tp = male_fair_t[1], test_female_tn = female_fair_t[0], test_female_tp = female_fair_t[1],
            #                 best_test_auroc=test_auroc, best_test_fpr=test_fpr, best_test_tpr=test_tpr)


            end_t = time.time()
            train_t = int((end_t - start_t) / 60)
            print('Training Time : {} hours {} minutes / iter : {}/{}'.format(int(train_t / 60), (train_t % 60),
                                                                              (iter_ + 1), n_iters))

            # get statistics
            Y_pred_train, Y_train, S_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                                    n_workers=self.n_workers, model=model)

            # calculate violation
            if self.reweighting_target_criterion == 'eo':
                acc, violations = self.get_error_and_violations_EO(Y_pred_train, Y_train, S_train, self.num_groups)
            elif self.reweighting_target_criterion == 'eopp':
                acc, violations = self.get_error_and_violations_Eopp(Y_pred_train, Y_train, S_train, self.num_groups)

            self.extended_multipliers =- self.eta * violations
            self.weight_matrix = self.get_weight_matrix(self.extended_multipliers, self.reweighting_target_criterion) 
            print(self.weight_matrix)
        
        # get last model results
        model = load_model(model, self.save_dir, self.log_name, load_best=False)

        _, eval_accs, roc_scores, eval_f1, acc_male_v, acc_female_v, male_fair_v, female_fair_v = self.evaluate(model, val_loader, self.criterion, self.cuda)
        val_acc, val_bal_acc = eval_accs
        val_auroc, val_fpr, val_tpr = roc_scores
        val_micro_F1, val_macro_F1 = eval_f1
        val_eo = max(abs(male_fair_v[0] - female_fair_v[0]), abs(male_fair_v[1] - female_fair_v[1]))
        val_eopp = abs(male_fair_v[1] - female_fair_v[1])

        _, eval_accs, roc_scores, eval_f1, acc_male_t, acc_female_t, male_fair_t, female_fair_t = self.evaluate(model, test_loader, self.criterion, self.cuda)
        test_acc, test_bal_acc = eval_accs
        test_auroc, test_fpr, test_tpr = roc_scores
        test_micro_F1, test_macro_F1 = eval_f1

        # last!!!
        self._update_log(logger=logger, is_last=True,
                         last_val_acc=val_acc, last_val_bal_acc=val_bal_acc, last_val_acc_male=acc_male_v, last_val_acc_female=acc_female_v,
                         last_val_male_tn = male_fair_v[0], last_val_male_tp = male_fair_v[1], last_val_female_tn = female_fair_v[0], last_val_female_tp = female_fair_v[1],
                         last_val_auroc=val_auroc, last_val_fpr=val_fpr, last_val_tpr=val_tpr,
                         last_val_eo=val_eo, last_val_eopp=val_eopp,
                         last_test_acc=test_acc, last_test_bal_acc=test_bal_acc, last_test_acc_male=acc_male_t, last_test_acc_female=acc_female_t,
                         last_test_male_tn = male_fair_t[0], last_test_male_tp = male_fair_t[1], last_test_female_tn = female_fair_t[0], last_test_female_tp = female_fair_t[1],
                         last_test_auroc=test_auroc, last_test_fpr=test_fpr, last_test_tpr=test_tpr)

        print('Training Finished!')
            # return model, logger

    def _train_epoch(self, model, epoch, epochs, train_loader, optimizer, scaler):        
        model = self._set_mode(model, eval=False)

        running_acc = 0.0
        running_loss = 0.0

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_data = 0

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            optimizer.zero_grad(set_to_none=True)

            # Get the inputs
            inputs, labels, sub_id, idx, groups, _ = self._process_data_for_train(data, self.cuda, self.binary) #groups = gender [F/M]

            # labels = labels.float() if num_classes == 2 else labels.long()
            # groups = groups.long()
            # labels = labels.long()

            if self.weight_matrix != None :
                weights = self.weight_matrix[groups.long(), labels.long()]
                if self.cuda:
                    weights = weights.cuda()
            

            with autocast(enabled=True):

                outputs = model(inputs)
                if (self.with_mci and outputs.dim() == 1) or len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)  # for only one data
                
                loss = self.criterion(outputs, labels)
                if self.weight_matrix != None :
                    loss = loss * weights
                loss = torch.sum(loss)

            running_loss += loss.detach()
            acc = get_accuracy(outputs, labels, binary=self.binary)
            running_acc += acc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach() * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.3f} '
                      '[{:.3f} s/batch]'.format
                      (epoch + 1, epochs, i + 1, 'LBC', running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = 0.0

        # last_batch_idx = i
        return epoch_loss.item() / num_data , epoch_acc.item() / num_data , scaler

    def get_statistics(self, dataset, batch_size, n_workers, model=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True, drop_last=False)

        if model != None:
            # model.eval()
            model = self._set_mode(model, eval=True)

        Y_pred_set = []
        Y_set = []
        S_set = [] # sensitive attribute

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # inputs, _, groups, labels, _ = data
                inputs, labels, sub_id, idx, groups, _ = self._process_data_for_train(data, self.cuda, self.binary) # groups = gender [F/M]

                Y_set.append(labels) # groups = -1 means no supervision for sensitive group
                S_set.append(groups)
                
                if model != None:
                    outputs = model(inputs)
                    Y_pred_set.append(torch.argmax(outputs, dim=1) if self.num_classes > 2 else (torch.sigmoid(outputs) >= 0.5).float())
                    # Y_pred_set.append(torch.argmax(outputs, dim=1))

        Y_set = torch.cat(Y_set)
        S_set = torch.cat(S_set)
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        return Y_pred_set.long(), Y_set.long().cuda(), S_set.long().cuda()

    def get_weight_matrix(self, extended_multipliers, target_criterion):  
        if target_criterion =='eopp': 
            w_matrix = torch.stack([1-torch.sigmoid(extended_multipliers), torch.sigmoid(extended_multipliers)]) # g by 0/1
        else : # eo
            w_matrix = torch.sigmoid(extended_multipliers.reshape(-1,2)) # g by F/T
        return w_matrix

   # Binarized version for Eopp
    def get_error_and_violations_Eopp(self, binarized_y_pred, binarized_y, groups, num_groups):
        binarized_acc = torch.mean((binarized_y_pred == binarized_y).float())
        violations = torch.zeros((num_groups,))
        for g in range(num_groups):
            protected_idxs = torch.where(torch.logical_and(groups == g, binarized_y > 0))
            positive_idxs = torch.where(binarized_y > 0)
            violations[g] = torch.mean(binarized_y_pred[protected_idxs].float()) - torch.mean(binarized_y_pred[positive_idxs].float())
        return binarized_acc, violations

    # Binarized version for EO
    def get_error_and_violations_EO(self, binarized_y_pred, binarized_y, groups, num_groups):
        binarized_acc = torch.mean((binarized_y_pred == binarized_y).float())
        violations = torch.zeros((num_groups*2,))
        for g in range(num_groups):
            protected_positive_idxs = torch.where(torch.logical_and(groups == g, binarized_y > 0))
            positive_idxs = torch.where(binarized_y > 0)
            violations[2*g+1] = torch.mean(binarized_y_pred[protected_positive_idxs].float()) - torch.mean(binarized_y_pred[positive_idxs].float())  # T
            protected_negative_idxs = torch.where(torch.logical_and(groups == g, binarized_y < 1))
            negative_idxs = torch.where(binarized_y < 1)
            violations[2*g] = torch.mean(binarized_y_pred[protected_negative_idxs].float()) - torch.mean(binarized_y_pred[negative_idxs].float())  # F
        return binarized_acc, violations

#    # Binarized version for Eopp
#     def get_error_and_violations_Eopp(self, binarized_y_pred, binarized_y, groups, num_groups):
#         binarized_acc = torch.mean((binarized_y_pred == binarized_y).float())
#         violations = torch.zeros((num_groups,))
#         for g in range(num_groups):
#             protected_idxs = torch.where(torch.logical_and(groups == g, binarized_y > 0))
#             positive_idxs = torch.where(binarized_y > 0)
#             violations[g] = torch.mean(binarized_y_pred[protected_idxs].float()) - torch.mean(binarized_y_pred[positive_idxs].float())
#         return binarized_acc, violations

#     # Binarized version for EO
#     def get_error_and_violations_EO(self, binarized_y_pred, binarized_y, groups, num_groups):
#         binarized_acc = torch.mean((binarized_y_pred == binarized_y).float())
#         violations = torch.zeros((num_groups*2,))
#         for g in range(num_groups):
#             protected_positive_idxs = torch.where(torch.logical_and(groups == g, binarized_y > 0))
#             positive_idxs = torch.where(binarized_y > 0)
#             violations[2*g+1] = torch.mean(binarized_y_pred[protected_positive_idxs].float()) - torch.mean(binarized_y_pred[positive_idxs].float())  # T
#             protected_negative_idxs = torch.where(torch.logical_and(groups == g, binarized_y < 1))
#             negative_idxs = torch.where(binarized_y < 1)
#             violations[2*g] = torch.mean(binarized_y_pred[protected_negative_idxs].float()) - torch.mean(binarized_y_pred[negative_idxs].float())  # F
#         return binarized_acc, violations


    # # Vectorized version for DP & multi-class
    # def get_error_and_violations_Eopp(self, y_pred, label, groups, num_groups, num_classes):
    #     acc = torch.mean((y_pred == label).float())
    #     total_num = len(y_pred)
    #     violations = torch.zeros((num_groups, num_classes))
    #     for g in range(num_groups):
    #         for c in range(num_classes):
    #             pivot = len(torch.where(y_pred==c)[0])/total_num
    #             group_idxs=torch.where(groups == g)[0]
    #             group_pred_idxs = torch.where(torch.logical_and(groups == g, y_pred == c))[0]
    #             violations[g, c] = len(group_pred_idxs)/len(group_idxs) - pivot
    #     return acc, violations

    # # Vectorized version for EO & multi-class
    # def get_error_and_violations_EO(self, y_pred, label, groups, num_groups, num_classes):
    #     acc = torch.mean((y_pred == label).float())
    #     violations = torch.zeros((num_groups, num_classes)) 
    #     for g in range(num_groups):
    #         for c in range(num_classes):
    #             class_idxs = torch.where(label==c)[0]
    #             pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
    #             pivot = len(pred_class_idxs)/len(class_idxs)
    #             group_class_idxs=torch.where(torch.logical_and(groups == g, label == c))[0]
    #             group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(groups == g, y_pred == c), label == c))[0]
    #             violations[g, c] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
    #     # print('violations',violations)
    #     return acc, violations    
    
    # def initialize_all(self):
    #     from networks import ModelFactory
    #     self.model = ModelFactory.get_model('lr', hidden_dim=64, num_classes=2, n_layer = 1)
    #     self.optimizer =optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #     self.scheduler = MultiStepLR(self.optimizer, [10,20], gamma=0.1)