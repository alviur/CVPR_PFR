import numpy as np
import torch
from copy import deepcopy
from argparse import ArgumentParser

from datasets.exemplars_dataset import ExemplarsDataset
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """
        LwF + Entropy based weighting
    """

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
        lamb=1.0,
        alpha=1.0,
        beta=1.0,
        T=2,
        log_debug=False
    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd, multi_softmax,
            wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset
        )

        self.model_old = None
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.log_debug = log_debug
        self._step = 0

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Page 5: "λ is a loss balance weight, set to 1 for most our experiments. Making λ larger will favor the old
        #  task performance over the new task’s, so we can obtain a old-task-new-task performance line by changing λ."
        parser.add_argument('--lamb', default=1, type=float, required=False)
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations."
        parser.add_argument('--T', default=2, type=int, required=False)
        # Entropy based weighting
        parser.add_argument('--alpha', default=1.0, type=float, required=False)
        parser.add_argument('--beta', default=1.0, type=float, required=False)
        parser.add_argument('--log_debug', action='store_true')
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
        # log number of task classes
        self.logger.tbwriter.add_scalar('overall/classes_per_taks', len(np.unique(trn_loader.dataset.labels)), t)
        self.logger.tbwriter.add_scalar('overall/train_samples', len(trn_loader.dataset), t)
        self.logger.tbwriter.add_scalar('overall/val_samples', len(val_loader.dataset), t)

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory
            )

        # FINETUNE TRAINING -- contains the epochs loop
        self._step = 0
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        total_loss = 0.0
        total_examples = 0
        for images, targets in trn_loader:
            self._step += 1
            # Forward current model
            outputs, embeddings = self.model(images.to(self.device), return_features=True)
            # Log embedding l2-norm
            self.logger.tbwriter.add_histogram(f"t{t}/emb_l2_norm_dist", torch.norm(embeddings, dim=1), self._step)
            # Forward old model
            targets_old = None
            if t > 0 and (self.lamb > 0 or self.alpha > 0):
                targets_old, embeddings_old = self.model_old(images.to(self.device), return_features=True)
                # log displacement
                self.logger.tbwriter.add_histogram(
                    f"t{t}/displacement_dist", (embeddings_old - embeddings).pow(2).sum(1).pow(0.5), self._step
                )
                self.logger.tbwriter.add_histogram(
                    f"t{t}/displacement_cos_dist", torch.nn.functional.cosine_similarity(embeddings, embeddings_old),
                    self._step
                )
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
                if t > 0 and (self.lamb > 0 or self.alpha > 0):
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(
        self, outputs, targets, exp=1.0, size_average=True, eps=1e-5, softmax_outputs=True, softmax_targets=True
    ):
        if softmax_outputs:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        if softmax_targets:
            targets = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            outputs = outputs.pow(exp)
            outputs = outputs / outputs.sum(1).view(-1, 1).expand_as(outputs)
            targets = targets.pow(exp)
            targets = targets / targets.sum(1).view(-1, 1).expand_as(targets)
        outputs = outputs + eps / outputs.size(1)
        outputs = outputs / outputs.sum(1).view(-1, 1).expand_as(outputs)
        ce = -(targets * outputs.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def entropy(self, outputs, eps=1e-5, softmax_outputs=True):
        if softmax_outputs:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs + eps / outputs.size(1)
        # outputs = outputs / outputs.sum(1).view(-1, 1).expand_as(outputs)
        entropy = -(outputs * outputs.log()).sum(1)
        return entropy

    # Returns the loss value
    def criterion(self, t, outputs, targets, targets_old):
        loss = 0
        if t > 0 and (self.lamb > 0 or self.alpha > 0):
            # Knowledge distillation loss for all previous tasks
            kd_with_temp = self.cross_entropy(
                torch.cat(outputs[:t], dim=1), torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T, size_average=False
            )

            # Log KD
            kd_with_temp_mean = kd_with_temp.mean()
            kd = self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(targets_old[:t], dim=1), size_average=False)
            kd_mean = kd.mean()
            if self.log_debug:
                self.logger.tbwriter.add_histogram(f"t{t}/loss/kd_with_temp_dist", kd_with_temp, self._step)
                self.logger.tbwriter.add_histogram(f"t{t}/loss/kd_dist", kd_with_temp_mean, self._step)

            self.logger.tbwriter.add_scalar(f"t{t}/loss/old_kd", kd_mean, self._step)
            self.logger.tbwriter.add_scalar(f"t{t}/loss/old_kd_with_temp", kd_with_temp_mean, self._step)

            ent_old = self.entropy(torch.cat(outputs[:t], dim=1))
            ent_old_model = self.entropy(torch.cat(targets_old[:t], dim=1))
            ent = self.entropy(outputs[t])

            if self.log_debug:
                self.logger.tbwriter.add_scalar(f"t{t}/ent_old_mean", ent_old.mean(), self._step)
                self.logger.tbwriter.add_scalar(f"t{t}/ent_old_model_mean", ent_old_model.mean(), self._step)
                self.logger.tbwriter.add_scalar(f"t{t}/ent_mean", ent.mean(), self._step)

            if self.lamb > 0:
                loss_kd = self.lamb * kd_with_temp_mean
                self.logger.tbwriter.add_scalar(f"t{t}/loss_kd", loss_kd, self._step)
                loss += loss_kd

            if self.alpha > 0:
                normal_base = self.model_old.task_cls.sum().float().cuda().log()
                w = ((normal_base - ent_old_model) / normal_base).pow(self.beta)
                wkd = (w * kd_with_temp).mean()
                loss_wkd = self.alpha * wkd
                self.logger.tbwriter.add_scalar(f"t{t}/loss_wkd", loss_wkd, self._step)
                loss += loss_wkd

        # Current cross-entropy loss
        if len(self.exemplars_dataset) > 0:
            # with exemplars use all heads
            ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets, reduction='none')
        else:
            ce = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t], reduction='none')

        if self.log_debug:
            self.logger.tbwriter.add_histogram(f"t{t}/loss/ce_dist", ce, self._step)
        ce = ce.mean()
        self.logger.tbwriter.add_scalar(f"t{t}/loss/ce", ce, self._step)
        loss += ce

        return loss
