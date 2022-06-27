from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
from pytorch_metric_learning import samplers, miners, losses
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader

from align_uniform_loss import AlignUniformLoss
from datasets.exemplars_selection import override_dataset_transform
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """ Class implementing the finetuning baseline """
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
        m_per_class=8,
        metric_loss='triplet',
        optim='SGD',
        margin=0.0,
        mining=False,
        cov_full=False,
        uniformity_alpha=0.0,
        verbose=False,
        log_debug=False
    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd, multi_softmax,
            wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset
        )
        self.uniformity_alpha = uniformity_alpha
        self.cov_full = cov_full
        self.mining = mining
        self.margin = margin
        self.metric_loss = metric_loss
        self.optim = optim
        self.m_per_class = m_per_class
        self.verbose = verbose
        self.log_debug = log_debug

        self.exemplar_means = None
        self.exemplar_covs = []
        self.val_transform = None

        self.show_batch_size_info = True
        self.init_loss()

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--m_per_class', type=int, default=8)
        parser.add_argument('--optim', type=str, default='SGD')
        parser.add_argument(
            '--metric_loss',
            default='triplet',
            type=str,
            required=False,
            choices=['triplet', 'align_uniform', 'contrastive', 'multiSimilarity', 'nPairs', 'proxyNCA', 'proxyAnchor']
        )
        parser.add_argument('--margin', default=0.0, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--mining', action='store_true', help='(default=%(default)s)')
        parser.add_argument('--uniformity_alpha', default=0.0, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--log_debug', action='store_true')
        parser.add_argument('--verbose', action='store_true')

        return parser.parse_known_args(args)

    # Returns the optimizer
    def _get_optimizer(self):
        print(f'Getting {self.optim} optimizer')
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        elif self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def pre_train_process(self, t, trn_loader):
        # for warm-up train use cross-entropy classifier
        self.criterion = super().criterion
        super().pre_train_process(t, trn_loader)
        # after remove head and train with metric loss
        self.model.remove_all_heads()
        self.criterion = self.criterion_metric

    def train_loop(self, t, trn_loader, val_loader):
        self.val_transform = val_loader.dataset.transform
        self.current_trn_loader = trn_loader
        # set embedding dim
        with torch.no_grad():
            self.emb_dim = self.model.embedding(trn_loader.dataset[0][0].unsqueeze(0).to(self.device)).shape[-1]
            print(f"Embedding dimension: {self.emb_dim}")

        sampler = samplers.MPerClassSampler(trn_loader.dataset.labels, m=self.m_per_class)
        m_per_class_loader = DataLoader(
            trn_loader.dataset,
            batch_size=trn_loader.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory
        )

        super().train_loop(t, m_per_class_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        total_loss = 0.0
        total_examples = 0
        _trn_loader = tqdm.tqdm(trn_loader) if self.verbose else trn_loader
        for images, targets in _trn_loader:
            if self.show_batch_size_info:
                self.show_batch_size_info = False
                print(f"Batch size: {len(targets)}")
            # Forward current model
            outputs = self.model.embedding(images.to(self.device))
            mask = outputs.isnan().sum(dim=1) == 0
            correct_embeddings_num = mask.sum().item()
            if correct_embeddings_num != len(targets):
                print(f'{len(targets) - correct_embeddings_num} images returned NaN while training')
            outputs = outputs[mask]
            targets = targets.to(self.device)[mask]
            # if targets.size() == 0:
            #     continue
            loss = self.criterion(t, outputs, targets)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)

        self.update_prototypes()
        return total_loss / total_examples

    def criterion_metric(self, t, outputs, targets):
        hard_pairs = self.miner(outputs, targets) if self.miner else None
        metric_loss = self.metric_loss_func(outputs, targets, hard_pairs)
        loss = metric_loss
        if self.uniformity_alpha > 0.0:
            temp = 2.0  # default value
            uniformity_loss = torch.pdist(outputs, p=2).pow(2).mul(-temp).exp().mean().log()
            loss += uniformity_loss
            print(f'Metric loss: {metric_loss} uniformity loss: {uniformity_loss}')
        # assert not torch.isnan(loss)
        # assert not torch.isinf(loss)
        return loss

    def update_prototypes(self):
        saved_prototypes = self.exemplar_means.size(0) if self.exemplar_means is not None else 0
        all_prototypes = self.model.task_cls.sum().item()
        if saved_prototypes != all_prototypes:
            print(f"Expanding prototypes from: {saved_prototypes} to {all_prototypes}")
            assert np.min(self.current_trn_loader.dataset.labels) == saved_prototypes  # continuity in labels
            new_prototypes = (all_prototypes - saved_prototypes)
            if self.exemplar_means is None:
                self.exemplar_means = torch.zeros((new_prototypes, self.emb_dim), device=self.device)
            else:
                self.exemplar_means = torch.cat(
                    (self.exemplar_means, torch.zeros((new_prototypes, self.emb_dim), device=self.device))
                )

        # print(f'Calculating prototypes for classes: {np.unique(self.current_trn_loader.dataset.labels)}')
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.current_trn_loader.dataset, self.val_transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep the same order
            _loader = DataLoader(
                _ds,
                batch_size=self.current_trn_loader.batch_size,
                shuffle=False,
                num_workers=self.current_trn_loader.num_workers,
                pin_memory=self.current_trn_loader.pin_memory
            )
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in _loader:
                    feats = self.model.embedding(images.to(self.device))
                    extracted_features.append(feats)
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # print(f'Class: {curr_cls} prototype calculation...')
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means[curr_cls] = cls_feats_mean

            for p in self.exemplar_means:
                assert p.sum().item() != 0

    # NMC classifier evaluation
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.to(self.device)
                outputs = self.model.embedding(images.to(self.device))
                mask = outputs.isnan().sum(dim=1) == 0
                if mask.sum().item() != len(targets):
                    print(f'{len(targets) - len(mask)} images returned NaN while evaluation')
                outputs = outputs[mask]
                targets = targets.to(self.device)[mask]
                loss = self.criterion(t, outputs, targets.to(self.device))
                # during training, the usual accuracy is computed on the outputs
                hits_taw, hits_tag = self.classify(t, outputs, targets)
                total_acc_taw += hits_taw
                total_acc_tag += hits_tag
                # Log
                total_loss += loss.item() * len(targets)
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Algorithm 1: iCaRL Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack([self.exemplar_means] * features.shape[0])
        means = means.transpose(1, 2)
        # features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        if self.cov_full:
            W = []
            for i in range(means.shape[2]):
                mean = means[0, :, i].cpu().numpy()
                cov = self.exemplar_covs[i]
                W_tmp = multivariate_normal.pdf(features[:, :, 0].cpu(), mean=mean, cov=cov)
                W.append(W_tmp)
            dists = -np.asarray(W).T
        else:
            dists = (features - means).pow(2).sum(1).squeeze()
            dists = dists.cpu().numpy()

        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task].numpy()
        offset = self.model.task_offset[task].numpy()
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.cpu().numpy()).sum().item()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.cpu().numpy()).sum().item()
        return hits_taw, hits_tag

    def init_loss(self):
        if self.mining:
            # Set the mining function
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        else:
            self.miner = None

        if self.metric_loss == 'triplet':
            self.metric_loss_func = losses.TripletMarginLoss(margin=self.margin)
        elif self.metric_loss == 'contrastive':
            self.metric_loss_func = losses.ContrastiveLoss(pos_margin=self.margin)
        elif self.metric_loss == 'multiSimilarity':
            self.metric_loss_func = losses.MultiSimilarityLoss(alpha=2.0, beta=40.0)
        elif self.metric_loss == 'nPairs':
            self.metric_loss_func = losses.NPairsLoss()
        # elif self.metric_loss == 'proxyNCA':
        #     loss_func = losses.ProxyNCALoss(num_classes = 100, embedding_size = self.model.out_size)
        # elif self.metric_loss == 'proxyAnchor':
        #     loss_func = losses.ProxyAnchorLoss(num_classes = 100, embedding_size = self.model.out_size, margin = 0.1, alpha = 32)
        elif self.metric_loss == 'align_uniform':
            self.metric_loss_func = AlignUniformLoss()
        else:
            raise RuntimeError(f"Unknown metric loss: {self.metric_loss}")
