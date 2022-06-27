from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from pytorch_metric_learning import losses, miners, samplers
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """
    LLL with Deep Metric Learning using pytorch_metric_learning library.
    """
    def __init__(self,
                 model,
                 device,
                 nepochs=60,
                 lr=0.5,
                 lr_min=1e-4,
                 lr_factor=3,
                 lr_patience=5,
                 clipgrad=10000,
                 lr_finetuning=0.01,
                 nepochs_finetuning=40,
                 momentum=0.9,
                 wd=0.00001,
                 logger=None,
                 exemplars_dataset=None,
                 metric_lambda=0.0,
                 classifier_lambda=1.0,
                 margin=0.0,
                 mining=False,
                 balanced=False,
                 icarl_lambda=1.0,
                 sampler=None,
                 eval_method='NMC',
                 optimizer='SGD',
                 fix_bn=False,
                 separate_head_training=False,
                 binary_ce=False,
                 metric_loss='triplet',
                 extra_head=False,
                 cov_full=False,
                 multi_softmax=False,
                 wu_nepochs=0,
                 wu_lr_factor=1,
                 eval_on_train=False):
        super(Appr,
              self).__init__(model, device, nepochs, lr, lr_min, lr_factor,
                             lr_patience, clipgrad, momentum, wd,
                             multi_softmax, wu_nepochs, wu_lr_factor,
                             fix_bn, eval_on_train, logger, exemplars_dataset)

        self.metric_lambda = metric_lambda
        self.classifier_lambda = classifier_lambda
        self.margin = margin
        self.model_old = None
        self.lr_finetuning = lr_finetuning
        self.nepochs_finetuning = nepochs_finetuning
        self.mining = mining
        self.balanced = balanced
        self.icarl_lambda = icarl_lambda
        self.sampler = sampler
        self.eval_method = eval_method
        self.optimizer_method = optimizer
        self.separate_head_training = separate_head_training
        self.binary_ce = binary_ce
        self.metric_loss = metric_loss
        self.extra_head = extra_head
        self.cov_full = cov_full

        self.exemplar_means = []

        if self.extra_head:
            self.model.extra_head = torch.nn.Linear(
                self.model.out_size, self.model.out_size).to(
                    self.device)  # Extra head for metric learning

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args): 
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to store up to K=2000 exemplars."
        parser.add_argument('--metric_lambda',
                            default=0.0,
                            type=float,
                            required=False,
                            help='(default=%(default)s)')
        parser.add_argument('--icarl_lambda',
                            default=1.0,
                            type=float,
                            required=False,
                            help='(default=%(default)s)')
        parser.add_argument('--classifier_lambda',
                            default=1.0,
                            type=float,
                            required=False,
                            help='(default=%(default)s)')
        parser.add_argument(
            '--eval_method',
            default='NMC',
            type=str,
            required=False,
            help='(default=%(default)s)',
            choices=['NMC', 'softmax', 'multiSoftmax', 'taskClassifier'])
        parser.add_argument('--metric_loss',
                            default='triplet',
                            type=str,
                            required=False,
                            help='(default=%(default)s)',
                            choices=[
                                'triplet', 'contrastive', 'multiSimilarity',
                                'nPairs', 'proxyNCA', 'proxyAnchor'
                            ])
        parser.add_argument('--optimizer',
                            default='SGD',
                            type=str,
                            required=False,
                            help='(default=%(default)s)')
        parser.add_argument('--margin',
                            default=0.0,
                            type=float,
                            required=False,
                            help='(default=%(default)s)')
        parser.add_argument('--mining',
                            action='store_true',
                            help='(default=%(default)s)')
        parser.add_argument('--extra_head',
                            action='store_true',
                            help='(default=%(default)s)')
        parser.add_argument('--separate_head_training',
                            action='store_true',
                            help='(default=%(default)s)')
        parser.add_argument('--binary_ce',
                            action='store_true',
                            help='(default=%(default)s)')
        parser.add_argument('--cov_full',
                            action='store_true',
                            help='(default=%(default)s)')
        parser.add_argument('--balanced',
                            action='store_true',
                            help='(default=%(default)s)')
        parser.add_argument(
            '--sampler',
            default='randomSampler',
            help='(default=%(default)s)',
            choices=['randomSampler', 'weightedSampler', 'mPerClassSampler'])
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr_finetuning',
                            default=0.01,
                            type=float,
                            required=False,
                            help='(default=%(default)s)')
        parser.add_argument('--nepochs_finetuning',
                            default=40,
                            type=int,
                            required=False,
                            help='(default=%(default)s)')
        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):
        if self.optimizer_method == 'SGD':
            return torch.optim.SGD(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.wd,
                                   momentum=self.momentum)
        elif self.optimizer_method == 'Adam':
            return torch.optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.wd)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2

        if self.eval_method == 'taskClassifier':
            self.model.task_classifer = torch.nn.Linear(
                self.model.out_size,
                t + 1).to(self.device)  # Task classification head

        self.exemplar_means = []
        if self.sampler == 'mPerClassSampler':
            self.MPerClassTrain(t, trn_loader, val_loader)
            if self.balanced:
                self.MPerClassTrain(t, trn_loader, val_loader, balanced=True)
        else:
            if t == 0:  # First task is simple training
                super().train_loop(t, trn_loader, val_loader)
            else:
                self._train_unbalanced(t, trn_loader, val_loader)
                if self.balanced:
                    self._train_balanced(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        ds = trn_loader.dataset if t == 0 else self.exemplars_dataset + trn_loader.dataset
        ex_loader = DataLoader(ds,
                               batch_size=trn_loader.batch_size,
                               shuffle=True,
                               num_workers=trn_loader.num_workers,
                               pin_memory=trn_loader.pin_memory)
        self.exemplars_dataset.collect_exemplars(self.model, ex_loader,
                                                 val_loader.dataset.transform)

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader,
                                       val_loader.dataset.transform)

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset,
                                        transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep the same order
            icarl_loader = DataLoader(_ds,
                                      batch_size=trn_loader.batch_size,
                                      shuffle=False,
                                      num_workers=trn_loader.num_workers,
                                      pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors, e.g. averages
            # are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device),
                                       return_features=True)[1]
                    # normalize
                    extracted_features.append(feats /
                                              feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)
                # self.exemplar_covs.append(np.cov(cls_feats.T.cpu().numpy()))

    # --- SIMILAR DISTILLATION APPROACH AS IN LwF --> see approaches/lwf.py --- modifications for iCaRL
    def post_train_process(self, t, trn_loader):
        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        if self.icarl_lambda > 0:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
            self.model_old.freeze_all()

    def MPerClassTrain(self, t, trn_loader, val_loader, balanced=False):
        if t == 0:
            ds = trn_loader.dataset
            targets = ds.labels
        else:
            exemplars_ds = self.exemplars_dataset
            trn_dataset = trn_loader.dataset
            if balanced:
                indices = torch.randperm(len(trn_dataset))
                trn_dataset = torch.utils.data.Subset(
                    trn_dataset, indices[:len(exemplars_ds)])

            ds = exemplars_ds + trn_dataset
            targets = np.hstack((exemplars_ds.labels, trn_dataset.labels))

        sampler = samplers.MPerClassSampler(targets, m=8)
        loader = DataLoader(ds,
                            batch_size=trn_loader.batch_size,
                            sampler=sampler,
                            drop_last=True,
                            num_workers=trn_loader.num_workers,
                            pin_memory=trn_loader.pin_memory)
        super().train_loop(t, loader, val_loader)

    def _train_unbalanced(self, t, trn_loader, val_loader):
        self._finetune_balanced = False
        loader = self._get_train_loader(trn_loader, False)
        super().train_loop(t, loader, val_loader)

    def _train_balanced(self, t, trn_loader, val_loader):
        self._finetune_balanced = True
        self.lr = self.lr_finetuning
        self.nepochs = self.nepochs_finetuning
        loader = self._get_train_loader(trn_loader, True)
        super().train_loop(t, loader, val_loader)

    def _get_train_loader(self, trn_loader, balanced=False):
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset,
                                                  indices[:len(exemplars_ds)])

        ds = exemplars_ds + trn_dataset

        if self.sampler == 'weightedSampler':
            # Get all targets
            targets = None
            for _, target in trn_loader:
                targets = target if targets is None else np.hstack(
                    (targets, target))
            targets = np.hstack((exemplars_ds.labels, targets))
            targets = torch.tensor(targets)
            # Compute samples weight (each sample should get its own weight)
            class_sample_count = torch.tensor([
                (targets == t).sum()
                for t in torch.unique(targets, sorted=True)
            ])
            weight = 1. / class_sample_count.float()
            samples_weight = torch.tensor([weight[t] for t in targets])
            # Create sampler, dataset, loader
            sampler = WeightedRandomSampler(samples_weight,
                                            len(samples_weight))

            return DataLoader(ds,
                              batch_size=trn_loader.batch_size,
                              sampler=sampler,
                              num_workers=trn_loader.num_workers,
                              pin_memory=trn_loader.pin_memory)

        elif self.sampler == 'randomSampler':
            return DataLoader(ds,
                              batch_size=trn_loader.batch_size,
                              shuffle=True,
                              num_workers=trn_loader.num_workers,
                              pin_memory=trn_loader.pin_memory)

    # from LwF: Runs a single epoch
    def train_epoch(self, t, trn_loader):
        if t > 0 and self.fix_bn:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        total_examples = 0

        for images, targets in trn_loader:
            # Forward old model
            outputs_old = None
            if t > 0 and self.icarl_lambda > 0:
                outputs_old = self.model_old(images.to(self.device))
            else:
                outputs_old = None
            # Forward current model
            outputs, embeddings = self.model(images.to(self.device),
                                             return_features=True)
            loss = self.classifier_lambda * self.criterion(
                t, outputs, targets.to(self.device), outputs_old)

            if self.metric_lambda > 0:
                if self.extra_head:
                    embeddings = self.model.extra_head(embeddings)
                loss += self.metric_lambda * self.criterion_metric(
                    t, embeddings, targets.to(self.device))

            if self.eval_method == 'taskClassifier' and t > 0:
                num_cls = self.model.task_cls[-1]
                loss += torch.nn.functional.cross_entropy(
                    self.model.task_classifer(embeddings),
                    (targets.to(self.device) //
                     num_cls))  # TODO: task ground truth
                loss_func = losses.TripletMarginLoss(margin=self.margin,
                                                     triplets_per_anchor="all")
                loss += loss_func(embeddings,
                                  (targets.to(self.device) // num_cls))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clipgrad)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)
        return total_loss / total_examples

    # from LwF: Contains the evaluation code
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_acc_task, total_num = 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.to(self.device)
                # Forward old model
                outputs_old = None
                if t > 0 and self.icarl_lambda > 0:
                    outputs_old = self.model_old(images)
                else:
                    outputs_old = None
                # Forward current model
                outputs, feats = self.model(images, return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device),
                                      outputs_old)
                if self.eval_method == 'taskClassifier':
                    output_task_classifier = self.model.task_classifer(feats)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw, hits_tag, hits_task = self.calculate_metrics(
                        outputs, targets)
                else:
                    if self.eval_method == 'NMC':
                        hits_taw, hits_tag, hits_task = self.classify(
                            t, feats, targets)
                    elif self.eval_method == 'softmax':
                        hits_taw, hits_tag, hits_task = self.calculate_metrics(
                            outputs, targets)
                    elif self.eval_method == 'multiSoftmax':
                        hits_taw, hits_tag, hits_task = self.multiSoftmax(
                            outputs, targets)
                    elif self.eval_method == 'taskClassifier':
                        hits_taw, hits_tag, hits_task = self.taskClassifier(
                            outputs, output_task_classifier, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_acc_task += hits_task.sum().item()
                total_num += len(targets)
            print('task accracy is: ' + str(total_acc_task / total_num))

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # classification and distillation terms from Alg. 3. -- original formulation has no trade-off parameter
    def criterion(self, t, outputs, targets, outputs_old):
        # Classification loss for new classes
        if self.binary_ce:
            targets_one_hot = torch.FloatTensor(
                targets.shape[0],
                len(outputs) * outputs[0].shape[1])
            targets_one_hot.zero_()
            targets_one_hot.scatter_(1, targets[:, None].cpu(), 1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.cat(outputs, dim=1), targets_one_hot.to(self.device))
        else:
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1),
                                                     targets)

        if self.separate_head_training:
            if t > 0:
                num_cls = self.model.task_cls[-1]
                loss = torch.zeros(1).to(self.device)
                task_labels = (targets // num_cls)
                for i in range(len(outputs)):
                    class_id = np.where(i == task_labels.cpu())[0]
                    if len(class_id) != 0:
                        if self.binary_ce:
                            targets_one_hot = torch.FloatTensor(
                                len(class_id), outputs[0].shape[1])
                            targets_one_hot.zero_()
                            targets_one_hot.scatter_(
                                1, targets[class_id, None].cpu() -
                                outputs[i].shape[1] * i, 1)
                            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                                outputs[i][class_id],
                                targets_one_hot.to(self.device))
                        else:
                            loss += torch.nn.functional.cross_entropy(
                                outputs[i][class_id],
                                (targets[class_id] - outputs[i].shape[1] * i))

        # Distilation loss for old classes
        if t > 0 and self.icarl_lambda > 0:
            # The original code does not match with the paper equation, but maybe sigmoid could be removed from g
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += sum(
                torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y])
                for y in range(sum(self.model.task_cls[:t])))
        return loss

    def criterion_metric(self, t, outputs, targets):
        if self.mining:
            # Set the mining function
            miner = miners.MultiSimilarityMiner(epsilon=0.1)
            hard_pairs = miner(outputs, targets)
        else:
            hard_pairs = None

        if self.metric_loss == 'triplet':
            loss_func = losses.TripletMarginLoss(margin=self.margin)
        elif self.metric_loss == 'contrastive':
            loss_func = losses.ContrastiveLoss(pos_margin=self.margin)
        elif self.metric_loss == 'multiSimilarity':
            loss_func = losses.MultiSimilarityLoss(alpha=2.0, beta=40.0)
        elif self.metric_loss == 'nPairs':
            loss_func = losses.NPairsLoss()
        elif self.metric_loss == 'proxyNCA':
            loss_func = losses.ProxyNCALoss(num_classes=100,
                                            embedding_size=self.model.out_size)
        elif self.metric_loss == 'proxyAnchor':
            loss_func = losses.ProxyAnchorLoss(
                num_classes=100,
                embedding_size=self.model.out_size,
                margin=0.1,
                alpha=32)
        else:
            raise RuntimeError("Bad metric loss function: {}".format(
                self.metric_loss))
        loss = loss_func(outputs, targets, hard_pairs)

        return loss

    # Algorithm 1: iCaRL Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        if self.cov_full:
            W = []
            for i in range(means.shape[2]):
                mean = means[0, :, i].cpu().numpy()
                cov = self.exemplar_covs[i]
                W_tmp = multivariate_normal.pdf(features[:, :, 0].cpu(),
                                                mean=mean,
                                                cov=cov)
                W.append(W_tmp)
            dists = -np.asarray(W).T
        else:
            dists = (features - means).pow(2).sum(1).squeeze()
            dists = dists.cpu().numpy()

        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task].numpy()
        offset = self.model.task_offset[task].numpy()
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.numpy()).sum()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.numpy()).sum()
        hits_task = (pred // num_cls == targets.numpy() // num_cls).sum()
        return hits_taw, hits_tag, hits_task

    # Contains the main Task-Aware and Task-Agnostic metrics
    def calculate_metrics(self, outputs, targets):
        # Task-Aware Multi-Head
        pred = torch.zeros_like(targets.to(self.device))
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax(
            ) + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        num_cls = self.model.task_cls[-1]
        pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        hits_task = (pred // num_cls == targets.to(self.device) //
                     num_cls).float()
        return hits_taw, hits_tag, hits_task

    # Contains the main Task-Aware and Task-Agnostic metrics
    def multiSoftmax(self, outputs, targets):
        # Task-Aware Multi-Head
        pred = torch.zeros_like(targets.to(self.device))
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax(
            ) + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        for head in range(len(outputs)):
            outputs[head] = torch.softmax(outputs[head], dim=1)
        pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        num_cls = self.model.task_cls[-1]
        hits_task = (pred // num_cls == targets.to(self.device) //
                     num_cls).float()
        return hits_taw, hits_tag, hits_task

    # Contains the main Task-Aware and Task-Agnostic metrics
    def taskClassifier(self, outputs, output_task_classifier, targets):
        # Task-Aware Multi-Head
        pred = torch.zeros_like(targets.to(self.device))
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax(
            ) + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        task_prediction = output_task_classifier.argmax(1)
        pred = [
            task_prediction[j] * outputs[0].shape[1] +
            outputs[task_prediction[j]][j].argmax()
            for j in range(len(targets))
        ]
        pred = torch.stack(pred)
        hits_tag = (pred == targets.to(self.device)).float()
        num_cls = self.model.task_cls[-1]
        hits_task = (task_prediction == targets.to(self.device) //
                     num_cls).float()
        return hits_taw, hits_tag, hits_task
