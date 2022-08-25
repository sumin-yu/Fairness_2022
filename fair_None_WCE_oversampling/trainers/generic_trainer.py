from __future__ import print_function

import time
import os
from utils import get_accuracy, get_auroc, save_model, load_model
from collections import defaultdict
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score


class GenericTrainer:
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.with_mci = args.with_mci
        self.cuda = args.cuda
        self.method = args.method
        self.num_classes = 3 if args.with_mci else 2
        self.binary = True if self.num_classes == 2 else False

        self.mode = args.mode
        self.model = None
        self._init_model(args)
        self.train_dataset, self.val_dataset, self.test_datsaet = self._get_dataset(args)
        self.train_loader, self.val_loader, self.test_loader = self._get_dataloader(args,
                                                                                    self.train_dataset,
                                                                                    self.val_dataset,
                                                                                    self.test_datsaet)

        pos_weight = torch.FloatTensor([args.pos_weight])
        if self.binary:
            if self.method == 'WCE':
                self.criterion = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda())
        else:
            # weight is not supported currently,
            # should be implemented if we want to consider multi-classes imbalance.
            self.criterion = nn.CrossEntropyLoss(weight=None)

        self.logger = defaultdict(list)
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.log_name = log_name
        self.term = args.term

    def _get_dataset(self, args):
        raise NotImplementedError

    def _get_dataloader(self, args, train_dataset, val_dataset, test_dataset, sampler=None):

        def _init_fn(worker_id):
            np.random.seed(int(self.seed))

        sampler, shuffle = None, True
        if args.method == 'oversampling':
            weights = train_dataset.make_weights(method=args.method)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                sampler=sampler, num_workers=args.num_workers, worker_init_fn=_init_fn,
                                pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, worker_init_fn=_init_fn,
                                        pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, worker_init_fn=_init_fn,
                                        pin_memory=True)
        
        return train_loader, val_loader, test_loader

    def _init_model(self, args):
        raise NotImplementedError

    def _set_mode(self, model, eval=False):
        if eval:
            model.eval()
        else:
            model.train()
        return model

    def _get_optim_n_scheduler(self, args, model):
        if isinstance(model, list):
            param_ms = []
            for m in model:
                param_ms.append(m.parameters())
            optimizer = optim.AdamW(param_ms, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=0.1)
        return optimizer, scheduler

    def _update_log(self, logger, is_last=False, **kwargs):
        for key, value in kwargs.items():
            logger[key].append(value)
        if is_last:
            suffix = "_log.pt"
            torch.save(logger, os.path.join(self.log_dir, self.log_name + suffix))

    def train(self, epochs, model=None, train_loader=None, val_loader=None, test_loader=None, logger=None):
        model = self.model if model is None else model
        train_loader = self.train_loader if train_loader is None else train_loader
        val_loader = self.val_loader if val_loader is None else val_loader
        test_loader = self.test_loader if test_loader is None else test_loader

        optimizer, scheduler = self._get_optim_n_scheduler(self.args, model)

        model = self._set_mode(model, eval=False)
        logger = self.logger if logger is None else logger
        best_acc, best_bal_acc = 0.0, 0.0

        scaler = GradScaler()
        for epoch in range(epochs):
            train_loss, train_acc, scaler , points = self._train_epoch(model, epoch, epochs, train_loader, optimizer, scaler)
            # m_ad,m_cn,f_ad,f_cn = points
            # print('m_ad {:.3f} m_cn {:.3f} f_ad {:.3f} f_cn {:.3f}'.format(m_ad/(m_ad+m_cn+f_ad+f_cn), m_cn/(m_ad+m_cn+f_ad+f_cn), f_ad/(m_ad+m_cn+f_ad+f_cn), f_cn/(m_ad+m_cn+f_ad+f_cn)))

            eval_start_time = time.time()
            val_loss, val_accs, val_roc_scores, val_f1, acc_male_v, acc_female_v, male_fair_v, female_fair_v = self.evaluate(
                model, val_loader, self.criterion, self.cuda
            )
            val_acc, val_bal_acc = val_accs
            val_micro_F1, val_macro_F1 = val_f1
            val_auroc, val_fpr, val_tpr = val_roc_scores
            val_eo = abs(male_fair_v[0] - female_fair_v[0]) + abs(male_fair_v[1] - female_fair_v[1])
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

            if self.binary:
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_model(model.state_dict(), self.save_dir, self.log_name, is_best=True)
            else:
                if val_bal_acc > best_bal_acc:
                    best_bal_acc = val_bal_acc
                    save_model(model.state_dict(), self.save_dir, self.log_name, is_best=True)

        save_model(model.state_dict(), self.save_dir, self.log_name)

        # get best model results
        model = load_model(model, self.save_dir, self.log_name, load_best=True)

        _, eval_accs, roc_scores, eval_f1, acc_male_v, acc_female_v, male_fair_v, female_fair_v = self.evaluate(model, val_loader, self.criterion, self.cuda)
        val_acc, val_bal_acc = eval_accs
        val_auroc, val_fpr, val_tpr = roc_scores
        val_micro_F1, val_macro_F1 = eval_f1
        val_eo = abs(male_fair_v[0] - female_fair_v[0]) + abs(male_fair_v[1] - female_fair_v[1])
        val_eopp = abs(male_fair_v[1] - female_fair_v[1])

        _, eval_accs, roc_scores, eval_f1, acc_male_t, acc_female_t, male_fair_t, female_fair_t = self.evaluate(model, test_loader, self.criterion, self.cuda)
        test_acc, test_bal_acc = eval_accs
        test_auroc, test_fpr, test_tpr = roc_scores
        test_micro_F1, test_macro_F1 = eval_f1

        self._update_log(logger=logger, is_last=True,
                         best_val_acc=val_acc, best_val_bal_acc=val_bal_acc, best_val_acc_male=acc_male_v, best_val_acc_female=acc_female_v,
                         best_val_male_tn = male_fair_v[0], best_val_male_tp = male_fair_v[1], best_val_female_tn = female_fair_v[0], best_val_female_tp = female_fair_v[1],
                         best_val_auroc=val_auroc, best_val_fpr=val_fpr, best_val_tpr=val_tpr,
                         best_val_eo=val_eo, best_val_eopp=val_eopp,
                         best_test_acc=test_acc, best_test_bal_acc=test_bal_acc, best_test_acc_male=acc_male_t, best_test_acc_female=acc_female_t,
                         test_male_tn = male_fair_t[0], test_male_tp = male_fair_t[1], test_female_tn = female_fair_t[0], test_female_tp = female_fair_t[1],
                         best_test_auroc=test_auroc, best_test_fpr=test_fpr, best_test_tpr=test_tpr)

        print('Training Finished!')
        return model, logger

    def _train_epoch(self, model, epoch, epochs, train_loader, optimizer, scaler):
        model = self._set_mode(model, eval=False)

        running_acc = 0.0
        running_loss = 0.0

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_data = 0
        
        ####oversampling####
        m_ad, m_cn, f_ad, f_cn = 0,0,0,0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)

            # Get the inputs
            inputs, labels, sub_id, idx, gender, weight = self._process_data_for_train(data, self.cuda, self.binary)

            ####oversampling####
            # for ii in range(4):
            #     if labels[ii]==0 and gender[ii]=='M': m_cn += 1
            #     elif labels[ii]==0 and gender[ii]=='F': f_cn += 1
            #     elif labels[ii]==1 and gender[ii]=='M': m_ad += 1
            #     else : f_ad += 1
            
            with autocast(enabled=True):

                outputs = model(inputs)
                if (self.with_mci and outputs.dim() == 1) or len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)  # for only one data

                loss = self.criterion(outputs, labels)
                if self.method == 'WCE':
                    loss = loss * weight.cuda()
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

            if i % self.term == 0:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print("[{}/{}, {:5d}] Train Loss: {:.3f} Train Acc: {:.3f} "
                      "[{:.3f} s/batch]".format
                      (epoch + 1, epochs, i, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        return epoch_loss.item() / num_data , epoch_acc.item() / num_data , scaler , (m_ad,m_cn,f_ad,f_cn)

    def _process_data_for_train(self, data, cuda=True, binary=True):
        # type conversion & to cuda
        if len(data) == 7: # 5->6
            inputs_0, inputs_1, labels, sub_id, idx, gender, weight = data
            input_0 = inputs_0.float()
            input_1 = inputs_1.float()
            if cuda:
                input_0 = input_0.cuda()
                input_1 = input_1.cuda()
            inputs = (input_0, input_1)
        else:
            inputs, labels, sub_id, idx, gender, weight = data
            inputs = inputs.float()
            if cuda:
                inputs = inputs.cuda()

        labels = labels.long() if not binary else labels.float()
        if cuda:
            labels = labels.cuda()
        return inputs, labels, sub_id, idx, gender, weight

    def evaluate(self, model, loader, criterion, cuda=True, logger = None):
        model = self._set_mode(model, eval=True)
        num_classes = loader.dataset.num_classes
        binary = num_classes == 2

        eval_acc = 0
        eval_loss = 0
        eval_data_count = 0

        eval_acc_male = 0 #
        eval_acc_female = 0 #
        eval_data_count_male = 0 #
        eval_data_count_female = 0 #
        # f1_score = 0
        acc_per_class = np.zeros(num_classes)
        num_per_class = np.zeros(num_classes)
        acc_per_class_male = np.zeros(num_classes) #
        acc_per_class_female = np.zeros(num_classes) #
        num_per_class_male = np.zeros(num_classes) #
        num_per_class_female = np.zeros(num_classes) #

        acc_per_gender = np.zeros(2) # [0] -> Male, [1] -> Female
        num_per_gender = np.zeros(2)

        roc_auc, fpr, tpr = None, None, None
        micro_f1, macro_f1 = None, None
        roc_auc_male, fpr_male, tpr_male = None, None, None #
        roc_auc_female, fpr_female, tpr_female = None, None, None #

        with torch.no_grad():
            yhats = []
            ys = []

            yhats_male = [] #
            ys_male = [] #
            yhats_female = [] #
            ys_female = [] #

            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, labels, sub_id, idx, gender, weight = self._process_data_for_train(eval_data, cuda, binary)
                outputs = model(inputs)



                if (self.with_mci and outputs.dim() == 1) or len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)  # for only one data

                outputs_male = outputs[np.where(np.array(gender)=='M')[0]] #
                outputs_female = outputs[np.where(np.array(gender)=='F')[0]] #
                labels_male = labels[np.where(np.array(gender)=='M')[0]] #
                labels_female = labels[np.where(np.array(gender)=='F')[0]] #

                loss = criterion(outputs, labels)
                if self.method == 'WCE':
                    loss = loss * weight.cuda()
                    loss = torch.sum(loss)

                ys.append(labels.cpu())
                ys_male.append(labels_male.cpu()) #
                ys_female.append(labels_female.cpu()) #

                eval_loss += loss * len(labels)


                yhats.append(outputs.cpu())
                yhats_male.append(outputs_male.cpu()) #
                yhats_female.append(outputs_female.cpu()) #

                hits = get_accuracy(outputs, labels, reduction="none", binary=binary)

                hits_male = get_accuracy(outputs_male, labels_male, reduction="none", binary=binary) #
                hits_female = get_accuracy(outputs_female, labels_female, reduction="none", binary=binary) #

                eval_acc += hits.sum()
                eval_data_count += len(labels)
                eval_acc_male += hits_male.sum()
                eval_acc_female += hits_female.sum()
                eval_data_count_male += len(labels_male)
                eval_data_count_female += len(labels_female)

                if hits.dim() == 0:
                    hits = hits.unsqueeze(0)
                if hits_male.dim() == 0:
                    hits_male = hits_male.unsqueeze(0)
                if hits_female.dim() ==0:
                    hits_female = hits_female.unsqueeze(0)

                for c in range(num_classes):
                    acc_per_class[c] += hits[(labels == c)].sum().data.cpu().numpy()
                    num_per_class[c] += (labels == c).sum().data.cpu().numpy()
                    acc_per_class_male[c] += hits_male[(labels_male==c)].sum().data.cpu().numpy()
                    acc_per_class_female[c] += hits_female[(labels_female==c)].sum().data.cpu().numpy()
                    num_per_class_male[c] += (labels_male == c).sum().data.cpu().numpy()
                    num_per_class_female[c] += (labels_female == c).sum().data.cpu().numpy()

            bal_acc = (acc_per_class / num_per_class).mean()
            male_acc_fair = acc_per_class_male / num_per_class_male #
            female_acc_fair = acc_per_class_female / num_per_class_female #

            eval_loss = eval_loss / eval_data_count
            
            eval_acc = eval_acc / eval_data_count
            eval_acc_male = eval_acc_male / eval_data_count_male
            eval_acc_female = eval_acc_female / eval_data_count_female

            if binary:
                roc_auc, fpr, tpr = get_auroc(yhats, ys)
                roc_auc_male, fpr_male, tpr_male = get_auroc(yhats_male, ys_male) #
                roc_auc_female, fpr_female, tpr_female = get_auroc(yhats_female, ys_female) #

            if self.mode == 'eval':        
                logger = self.logger if logger is None else logger
                
                eo = abs(male_acc_fair[0] - female_acc_fair[0]) + abs(male_acc_fair[1] - female_acc_fair[1])
                eopp = abs(male_acc_fair[1] - female_acc_fair[1])

                self._update_log(logger=logger, is_last=True,
                            acc=eval_acc.item(), bal_acc=bal_acc,
                            acc_male = eval_acc_male.item(), acc_female = eval_acc_female.item(),
                            male_tn = male_acc_fair[0], male_tp = male_acc_fair[1], female_tn = female_acc_fair[0], female_tp = female_acc_fair[1],
                            auroc=roc_auc, fpr=fpr, tpr=tpr,
                            eo=eo, eopp=eopp,
                            auroc_male=roc_auc_male, fpr_male=fpr_male, tpr_male=tpr_male,
                            auroc_female=roc_auc_female, fpr_female=fpr_female, tpr_female=tpr_female)
                return

            # else:
            return eval_loss.item(), (eval_acc.item(), bal_acc), (roc_auc, fpr, tpr), (micro_f1, macro_f1), eval_acc_male.item(), eval_acc_female.item(), (male_acc_fair[0], male_acc_fair[1]), (female_acc_fair[0], female_acc_fair[1])
