import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .learning_approach import Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Learning_Appr):
    """ Class implementing the joint baseline """
    def __init__(self,
                 model,
                 device,
                 nepochs=100,
                 lr=0.05,
                 lr_min=1e-4,
                 lr_factor=3,
                 lr_patience=5,
                 clipgrad=10000,
                 momentum=0,
                 wd=0,
                 multi_softmax=False,
                 wu_nepochs=0,
                 wu_lr_factor=1,
                 fix_bn=False,
                 eval_on_train=False,
                 logger=None,
                 exemplars_dataset=None,
                 task_freeze_features=-1):
        super(Appr,
              self).__init__(model, device, nepochs, lr, lr_min, lr_factor,
                             lr_patience, clipgrad, momentum, wd,
                             multi_softmax, wu_nepochs, wu_lr_factor, fix_bn,
                             eval_on_train, logger, exemplars_dataset)
        self.trn_datasets = []
        self.val_datasets = []
        self.task_freeze_features = task_freeze_features

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument(
            '--task_freeze_features',
            default=-1,
            type=int,
            required=False,
            help=
            'train Joint until specified task, then freeze features and continue Joint'
            '(-1: normal Joint, no freeze) (default=%(default)s)')
        return parser.parse_known_args(args)

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        if self.task_freeze_features > -1 and t >= self.task_freeze_features:
            self.model.freeze_all()
            for head in self.model.heads:
                for param in head.parameters():
                    param.requires_grad = True

    # Runs a single epoch
    def train_loop(self, t, trn_loader, val_loader):
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)

        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        if self.task_freeze_features < 0 or t <= self.task_freeze_features:
            self.model.train()
            if self.fix_bn and t > 0:
                self.model.freeze_bn()
        else:
            self.model.eval()
            for head in self.model.heads:
                head.train()
    
        total_loss = 0.0
        total_examples = 0
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clipgrad)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)
        return total_loss / total_examples

    # Returns the loss value
    def criterion(self, t, outputs, targets):
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1),
                                                 targets)


class JointDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
