import torch
from copy import deepcopy
from argparse import ArgumentParser

from datasets.exemplars_dataset import ExemplarsDataset
from .learning_approach import Learning_Appr


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Appr(Learning_Appr):
    """ Class implementing the Learning Without Forgetting approach
        described in https://arxiv.org/abs/1606.09282 """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None,
                 exemplars_dataset=None, lamb=1, T=2, kd_exemplars=False, ce_all_outputs=False, ce_exemplars_task_aware=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)

        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.kd_exemplars = kd_exemplars
        self.ce_all_outputs = ce_all_outputs
        self.ce_exemplars_task_aware = ce_exemplars_task_aware

        # data loaders for old/new balanced train
        self._train_dl_new = None
        self._train_dl_exemplars = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Page 5: "λ is a loss balance weight, set to 1 for most our experiments. Making λ larger will favor the old
        #  task performance over the new task’s, so we can obtain a old-task-new-task performance line by changing λ."
        parser.add_argument('--lamb', default=1, type=float, required=False, help='(default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations."
        parser.add_argument('--T', default=2, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--kd_exemplars', action='store_true')
        # parser.add_argument('--ce_all_outputs', action='store_true')
        # parser.add_argument('--ce_exemplars_task_aware', action='store_true')
        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            self._train_dl_new = torch.utils.data.DataLoader(trn_loader.dataset,
                                                     batch_size=trn_loader.batch_size//2,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
            self._train_dl_exemplars = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size//2,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
            
        # FINETUNE TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        if t > 0:
            return self.train_epoch_balanced(t)

        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        total_loss = 0.0
        total_examples = 0
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = torch.nn.functional.cross_entropy(outputs[t], targets.to(self.device) - self.model.task_offset[t])
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)

        return total_loss / total_examples

    # Runs a single epoch
    def train_epoch_balanced(self, t):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        total_loss = 0.0
        total_examples = 0
        for batch_new, batch_old in zip(self._train_dl_new, cycle(self._train_dl_exemplars)):
            images, new_targets = batch_new
            e_images, e_targets = batch_old
            
            
            new_outputs = self.model(images.to(self.device))
            new_outputs_old = self.model_old(images.to(self.device))
            e_outputs = self.model(e_images.to(self.device))
            
            # KD with new data
            loss = self.lamb * self.cross_entropy(torch.cat(new_outputs_old[:t], dim=1),
                                                    torch.cat(new_outputs[:t], dim=1), exp=1.0 / self.T)
            
            if not self.kd_exemplars:
                e_outputs_old = self.model_old(e_images.to(self.device))
                loss += self.lamb * self.cross_entropy(torch.cat(e_outputs_old[:t], dim=1),
                                                    torch.cat(e_outputs[:t], dim=1), exp=1.0 / self.T)
            
            # CE
            outputs = torch.cat([torch.cat(new_outputs, dim=1), torch.cat(e_outputs, dim=1)])
            targets = torch.cat([new_targets, e_targets])          
            loss += torch.nn.functional.cross_entropy(outputs, targets.to(self.device))

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
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                # loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                # total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
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
