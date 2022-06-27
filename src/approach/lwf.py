import torch
from copy import deepcopy
from argparse import ArgumentParser

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
        kd_only_new=False,
        ce_all_outputs=False,
        ce_exemplars_task_aware=False,
        ce_additional_current_task=False
    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd, multi_softmax,
            wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset
        )

        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.kd_only_new = kd_only_new
        self.ce_all_outputs = ce_all_outputs
        self.ce_exemplars_task_aware = ce_exemplars_task_aware
        self.ce_additional_current_task = ce_additional_current_task

        # internal vars
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
        parser.add_argument('--lamb', default=1, type=float, required=False, help='(default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations."
        parser.add_argument('--T', default=2, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--kd_only_new', action='store_true')
        parser.add_argument('--ce_all_outputs', action='store_true')
        parser.add_argument('--ce_exemplars_task_aware', action='store_true')
        parser.add_argument('--ce_additional_current_task', action='store_true')
        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and self.ce_all_outputs == False:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
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
        self._step = 1
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)
            self._step += 1

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
        if t > 0:
            kd_loss = 0
            if self.kd_only_new:
                new_data_indices = targets >= self.model.task_offset[t]
                if new_data_indices.any():
                    # Knowledge distillation loss for all previous tasks
                    # with only new data
                    kd_loss += self.lamb * cross_entropy(
                        torch.cat(outputs[:t], dim=1)[new_data_indices],
                        torch.cat(targets_old[:t], dim=1)[new_data_indices],
                        exp=1.0 / self.T
                    )
            else:
                # Knowledge distillation loss for all previous tasks
                kd_loss += self.lamb * cross_entropy(
                    torch.cat(outputs[:t], dim=1), torch.cat(targets_old[:t], dim=1), exp=1.0 / self.T
                )

            self.logger.tbwriter.add_scalar(f"t{t}/kd-loss", kd_loss.item(), self._step)
            loss += kd_loss

        # Current cross-entropy loss
        if len(self.exemplars_dataset) > 0 or self.ce_all_outputs:
            ce_current_exemplars = 0
            if self.ce_exemplars_task_aware:
                # with exemplars use all heads - but in task aware
                for _t in range(t + 1):
                    mask = targets >= self.model.task_offset[_t]
                    if _t < t:
                        mask &= targets < self.model.task_offset[_t + 1]
                    if mask.any():
                        ce_current_exemplars += torch.nn.functional.cross_entropy(
                            outputs[_t][mask], targets[mask] - self.model.task_offset[_t]
                        )
            else:
                # with exemplars use all heads like a single one
                ce_current_exemplars = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

            self.logger.tbwriter.add_scalar(f"t{t}/ce-current-exemplars", ce_current_exemplars.item(), self._step)
            loss += ce_current_exemplars

            # Additonal entropy on current head with current data
            if self.ce_additional_current_task:
                mask = targets >= self.model.task_offset[t]  # take only current data from mini-batch
                if mask.any():
                    ce_add_current = torch.nn.functional.cross_entropy(
                        outputs[t][mask], targets[mask] - self.model.task_offset[t]
                    )
                else:
                    ce_add_current = torch.float(0.0)
                self.logger.tbwriter.add_scalar(f"t{t}/ce-current-additional", ce_add_current.item(), self._step)
                loss += ce_add_current
        else:
            loss += torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

        return loss
