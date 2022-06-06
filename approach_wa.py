from copy import deepcopy
from torch.utils.data import DataLoader
from datasets.exemplars_dataset import ExemplarsDataset
import time
import torch
import numpy as np
from argparse import ArgumentParser
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm



class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05,momentum=0.9,
                 wd=0.0001,exemplars_dataset: ExemplarsDataset = None,milestone=None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.exemplars_dataset = exemplars_dataset
        self.optimizer = None
        self.milestone = milestone

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_loader, tst_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, tst_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def train_loop(self, t, trn_loader, tst_loader):
        """Contains the epochs loop"""
        self.optimizer = self._get_optimizer()
        scheduler = MultiStepLR(optimizer=self.optimizer, milestones=self.milestone, gamma=0.1)

        # Loop epochs
        for e in range(self.nepochs):
            print("-"*108)
            cur_lr = self.get_lr(self.optimizer)

            print('| Epoch %d, lr %.6f| training ... |'%(e,cur_lr))
            # Train
            self.train_epoch(t, trn_loader)
            scheduler.step()

            if e in self.milestone:
                train_acc = self.eval(t,trn_loader)
                self.eval(t,tst_loader)

        after_stage1_eval = self.eval(t,tst_loader)
        print("-"*108,"\n","***|after stage1 training, acc : %.4f "%(after_stage1_eval))

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        pass

    def eval(self, t, tst_loader):
        """Contains the evaluation code"""
        pass

    def calculate_metrics(self, outputs, targets):
        pass

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

class WA(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=250, lr=0.1,momentum=0.9, wd=0.0001,exemplars_dataset=None,milestone=None,
                 val_exemplar_percentage=0.1, num_bias_epochs=200, T=2, lamb=-1):
        super(WA, self).__init__(model, device, nepochs, lr, momentum, wd, exemplars_dataset,milestone)

        self.val_percentage = val_exemplar_percentage 
        self.bias_epochs = num_bias_epochs 
        self.model_old = None
        self.T = T 
        self.lamb = lamb

        self.num_exemplars = self.exemplars_dataset.max_num_exemplars  # 2000

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=-1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)') #-1默认使用文中的计算公式
        parser.add_argument('--T', default=2, type=int, required=False,help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, tst_loader):
        num_cls = sum(self.model.task_cls)
        num_old_cls = sum(self.model.task_cls[:t])

        if self.exemplars_dataset.max_num_exemplars != 0:
            num_exemplars_per_class = int(np.floor(self.num_exemplars / num_cls))

        # add exemplars to train_loader -- train_new + train_old (Fig.2)
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers)

        # STAGE 1: DISTILLATION
        print('Stage 1: Training model with distillation')
        super().train_loop(t, trn_loader, tst_loader)


        # STAGE 2: BIAS CORRECTION
        if t > 0:
            print('Stage 2: bias correction ...')
            old_weight_norms = []
            for i in range(t):
                old_weight_norm = torch.norm(self.model.heads[i].weight, p=2, dim=1)
                old_weight_norms.append(old_weight_norm)
            old_weight_norms = torch.concat(old_weight_norms,dim=0)
            new_weight_norm = torch.norm(self.model.heads[t].weight,p=2,dim=1)
            gamma = old_weight_norms.mean() / new_weight_norm.mean()
            self.model.heads[t].weight.data = gamma * self.model.heads[t].weight.data
            print("Experience %d , gamma : %.6f"%(t,gamma))

        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # STAGE 3: EXEMPLAR MANAGEMENT
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, tst_loader.dataset.transform)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        losses = []
        for images, targets in trn_loader:
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            losses.append(loss.item())
        print('train_loss : %.6f'%(np.mean(losses)))


    def eval(self, t, tst_loader):
        """Contains the evaluation code"""
        self.model.eval()
        with torch.no_grad():
            total_hits, total_num = 0, 0
            for images, targets in tst_loader:
                outputs = self.model(images.to(self.device))
                pred = torch.cat(outputs, dim=1).argmax(1)
                hits = (pred == targets.to(self.device)).float()
                total_hits += hits.sum().item()
                total_num += len(targets)
        return total_hits/total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, targets_old):
        """
        Returns the loss value
        """
        # Knowledge distillation loss for all previous tasks
        loss_dist = 0
        if t > 0:
            loss_dist += self.cross_entropy(torch.cat(outputs[:t], dim=1),torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T)
        # trade-off - the lambda from the paper if lamb=-1
        if self.lamb == -1:
            lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device)
            return (1.0 - lamb) * torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1),
                                                                    targets) + lamb * loss_dist  # (1-lamb) x Lc + lamb x Ld
        else:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + self.lamb * loss_dist
