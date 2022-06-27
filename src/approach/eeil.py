import torch
from copy import deepcopy
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .learning_approach import Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Learning_Appr):
    """
    Class implementing the End-to-end Incremental Learning (EEIL) approach described in:
        http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf
    Ref. code repository:
        https://github.com/fmcp/EndToEndIncrementalLearning
    Helpful code repo:
        https://github.com/arthurdouillard/incremental_learning.pytorch
    """

    def __init__(self, model, device, nepochs=90, lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, lamb=1.0, T=2, lr_finetuning_factor=0.1, nepochs_finetuning=40,
                 noise_grad=False, exemplars_dataset=None, unbalanced_no_exemplars=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)

        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad
        self.unbalanced_no_exemplars = unbalanced_no_exemplars

        self._train_epoch = 0
        self._finetune_balanced = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Added trade-off between the terms of Eq. 1 -- L = L_C + lamb * L_D
        parser.add_argument('--lamb', default=1.0, type=float, required=False, help='(default=%(default)s)')
        # "Based on our empirical results, we set T to 2 for all our experiments" (page 6)
        parser.add_argument('--T', default=2.0, type=float,
                            required=False, help='(default=%(default)s)')
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr_finetuning_factor', default=0.1, type=float,
                            required=False, help='(default=%(default)s)')
        # Number of epochs for balanced training
        parser.add_argument('--nepochs_finetuning', default=40, type=int,
                            required=False, help='(default=%(default)s)')
        # the addition of noise to the gradients
        parser.add_argument('--noise_grad', action='store_true', help='(default=%(default)s)')
        # don't use exemplars while doing unbalanced train
        parser.add_argument('--unbalanced_no_exemplars', action='store_true')
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        if t == 0:  # First task is simple training
            super().train_loop(t, trn_loader, val_loader)
            loader = trn_loader
        else:
            # Below procedure is described in paper, page 4:
            # "4. Incremental Learning"
            # Only modification is that instead of preparing examplars
            # before the training, we doing it online using old model.

            # Training process (new + old) - unbalanced training
            loader = self._train_unbalanced(t, trn_loader, val_loader)

            # Balanced fine-tunning (new + old)
            self._train_balanced(t, trn_loader, val_loader)

        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        # save model
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _train_unbalanced(self, t, trn_loader, val_loader):
        self._finetune_balanced = False
        self._train_epoch = 0
        if self.unbalanced_no_exemplars:
            loader = trn_loader
        else:
            loader = self._get_train_loader(trn_loader, False)
        super().train_loop(t, loader, val_loader)
        return loader

    def _train_balanced(self, t, trn_loader, val_loader):
        self._finetune_balanced = True
        self._train_epoch = 0
        orig_lr = self.lr
        self.lr *= self.lr_finetuning_factor
        orig_nepochs = self.nepochs
        self.nepochs = self.nepochs_finetuning
        loader = self._get_train_loader(trn_loader, True)
        super().train_loop(t, loader, val_loader)
        self.lr = orig_lr
        self.nepochs = orig_nepochs

    def _get_train_loader(self, trn_loader, balanced=False):
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset, indices[:len(exemplars_ds)])

        ds = exemplars_ds + trn_dataset
        return DataLoader(ds,
                          batch_size=trn_loader.batch_size,
                          shuffle=True,
                          num_workers=trn_loader.num_workers,
                          pin_memory=trn_loader.pin_memory)

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        total_loss = 0.0
        total_examples = 0
        for images, targets in trn_loader:
            images = images.to(self.device)
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # "We apply L2-regularization and random noise [21] (with parameters η = 0.3, γ = 0.55)
            # on the gradients to minimize overfitting" (page 8)
            # https://github.com/fmcp/EndToEndIncrementalLearning/blob/master/cnn_train_dag_exemplars.m#L367
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            if self.noise_grad:
                self._noise_grad(self.model.parameters(), self._train_epoch)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)

        self._train_epoch += 1
        return total_loss / total_examples

    # Returns the loss value -- original formulation has no trade-off parameter
    def criterion(self, t, outputs, targets, outputs_old=None):
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distilation loss
        if t > 0 and outputs_old:
            # take into account current head when balanced finetunning
            last_head_idx = t if self._finetune_balanced else (t - 1)
            for i in range(last_head_idx):
                loss += self.lamb * F.binary_cross_entropy(F.softmax(outputs[i] / self.T, dim=1),
                                                           F.softmax(outputs_old[i] / self.T, dim=1))
        return loss

    # Add noise to the gradients
    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)
