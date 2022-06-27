import torch
from copy import deepcopy
from argparse import ArgumentParser

from torch.utils.data.dataloader import DataLoader

from datasets.exemplars_dataset import ExemplarsDataset
from networks.loss import cross_entropy
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the Learning Without Forgetting approach
        described in https://arxiv.org/abs/1606.09282 """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(
        self,
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
        lamb=1,
        T=2,
        lr_finetuning_factor=0.1,
        nepochs_finetuning=40,
        only_head_finetuning=False
    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd, multi_softmax,
            wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset
        )

        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.only_head_finetuning = only_head_finetuning

        self._train_epoch = 0
        self._finetune_balanced = None
        self._t = None
        self._step = 0

    def _phase_name(self):
        return "2_balance" if self._finetune_balanced else "1_unbalanced"

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Page 5: "λ is a loss balance weight, set to 1 for most our experiments. Making λ larger will favor the old
        #  task performance over the new task’s, so we can obtain a old-task-new-task performance line by changing λ."
        parser.add_argument('--lamb', default=10, type=float, required=False, help='(default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations."
        parser.add_argument('--T', default=2, type=int, required=False, help='(default=%(default)s)')

        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr_finetuning_factor', default=0.1, type=float, required=False)
        # Number of epochs for balanced training
        parser.add_argument('--nepochs_finetuning', default=40, type=int, required=False)
        parser.add_argument('--only_head_finetuning', action='store_true')

        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):

        if self.only_head_finetuning and self._finetune_balanced:
            params = []
            for h in self.model.heads:
                params.extend(h.parameters())
        else:
            if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
                # if there are no exemplars, previous heads are not modified
                params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
            else:
                params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        self._t = t
        if t == 0:  # First task is simple training
            super().train_loop(t, trn_loader, val_loader)
            loader = trn_loader
        else:
            # Training process (new + old) - unbalanced training
            loader = self._train_unbalanced(t, trn_loader, val_loader)
            if len(self.exemplars_dataset) > 0:
                # Balanced fine-tunning (new + old)
                self._train_balanced(t, trn_loader, val_loader)

        # After task training update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        # save model
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _train_unbalanced(self, t, trn_loader, val_loader):
        self._step = 0
        self._finetune_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(trn_loader, False)
        super().train_loop(t, loader, val_loader)
        return loader

    def _train_balanced(self, t, trn_loader, val_loader):
        self._step = 0
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
        return DataLoader(
            ds,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory
        )

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        total_loss = 0.0
        total_examples = 0
        for images, targets in trn_loader:
            self._step += 1
            # Forward old model
            targets_old = None
            if t > 0 and not self._finetune_balanced:
                targets_old = self.model_old(images.to(self.device))  # distill on unbalanced
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)

        return total_loss / total_examples

    # Contains the evaluation code
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0 and not self._finetune_balanced:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Returns the loss value
    def criterion(self, t, outputs, targets, targets_old):
        loss = 0
        if t > 0 and targets_old is not None:
            # Knowledge distillation loss for all previous tasks
            loss_kd = self.lamb * cross_entropy(
                torch.cat(outputs[:t], dim=1), torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T
            )
            loss += loss_kd
            self.logger.tbwriter.add_scalar(f"t{t}/{self._phase_name()}-loss_kd", loss_kd, self._step)

        # Current cross-entropy loss
        loss_ce = 0
        if self._t > 0 and len(self.exemplars_dataset) > 0 and self._finetune_balanced:
            loss_ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            if t == 0:
                loss_ce = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
            else:
                new_data_indices = targets >= self.model.task_offset[t]
                if new_data_indices.any():
                    # Knowledge distillation loss for all previous tasks
                    # with only new data
                    loss_ce = torch.nn.functional.cross_entropy(
                        outputs[t][new_data_indices], targets[new_data_indices] - self.model.task_offset[t]
                    )

        loss += loss_ce
        self.logger.tbwriter.add_scalar(f"t{t}/{self._phase_name()}-loss_ce", loss_ce.item(), self._step)
        return loss
