import numpy as np
import torch
import math
from copy import deepcopy
from argparse import ArgumentParser
from torch import nn
import torch.nn
import torch.nn.functional as F
from torch.nn.functional import embedding

from datasets.exemplars_dataset import ExemplarsDataset
from .learning_approach import Learning_Appr


class Appr(Learning_Appr):
    """
        LwF + Entropy based weighting + Cosine similarity
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
        lambda_cos_base=5.0,
        log_debug=False,
        remove_adapt_lamda=False
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
        self.lambda_cos_base = lambda_cos_base
        self.log_debug = log_debug
        self.adapt_lamda = not remove_adapt_lamda
        self._step = 0

        self.lambda_cos = self.lambda_cos_base


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

        # Cos loss
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument(
            '--lambda_cos_base',
            default=0.,
            type=float,
            required=False,
        )
        parser.add_argument('--remove_adapt_lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
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

    def pre_train_process(self, t, trn_loader):
        # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
        # negative values"
        if t == 0:
            "network is not detected as ResNet and last ReLU needs to be removed!"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(
                    old_block.conv1, old_block.bn1, old_block.relu, old_block.conv2, old_block.bn2, old_block.downsample
                )
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        self.model.to(self.device)

        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma

            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.lambda_cos_base > 0:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True

            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lambda_cos = self.lambda_cos_base * math.sqrt(
                    sum([h.out_features for h in self.model.heads[:-1]]) / self.model.heads[-1].out_features
                )
                print('Adopting lambda: ', self.lambda_cos)

        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, I think that is not mentioned in the paper and doesn't seem to make a significance difference.
        # TODO: Add imprint weights (from original code)

        super().pre_train_process(t, trn_loader)

    # Runs after training all the epochs of the task (at the end of train function)
    def post_train_process(self, t, trn_loader):
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.model_old.heads:
            h.train()
        self.model_old.freeze_all()


    def _model_forward(self, model, inputs):
        r = model(inputs, return_features=True)
        # outputs = [_r[0] for _r in r]
        # embedding = r[-1][1]
        # return  outputs, embedding
        return r

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
            targets_old, embeddings_old = None, None
            if t > 0 and (self.lamb > 0 or self.alpha > 0 or self.lambda_cos_base > 0):
                targets_old, embeddings_old = self.model_old(images.to(self.device), return_features=True)
                # log displacement
                self.logger.tbwriter.add_histogram(
                    f"t{t}/displacement_dist", (embeddings_old - embeddings).pow(2).sum(1).pow(0.5), self._step
                )
                self.logger.tbwriter.add_histogram(
                    f"t{t}/displacement_cos_dist", torch.nn.functional.cosine_similarity(embeddings, embeddings_old),
                    self._step
                )
            loss = self.criterion(t, outputs, targets.to(self.device), targets_old, embeddings, embeddings_old)
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
                targets_old, emb_old = None, None
                if t > 0 and (self.lamb > 0 or self.alpha > 0 or self.lambda_cos_base > 0):
                    targets_old, emb_old = self.model_old(images.to(self.device), return_features=True)
                # Forward current model
                outputs, emb = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old, emb, emb_old)
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
    def criterion(self, t, outputs, targets, outputs_old=None, embeddings=None, embeddings_old=None):
        loss = 0
        if t > 0 and (self.lamb > 0 or self.alpha > 0 or self.lambda_cos > 0):
            # Knowledge distillation loss for all previous tasks
            kd_with_temp = self.cross_entropy(
                torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T, size_average=False
            )

            # Log KD
            kd_with_temp_mean = kd_with_temp.mean()
            kd = self.cross_entropy(
                torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1), size_average=False
            )
            kd_mean = kd.mean()
            if self.log_debug:
                self.logger.tbwriter.add_histogram(f"t{t}/loss/kd_with_temp_dist", kd_with_temp, self._step)
                self.logger.tbwriter.add_histogram(f"t{t}/loss/kd_dist", kd_with_temp_mean, self._step)

            self.logger.tbwriter.add_scalar(f"t{t}/loss/old_kd", kd_mean, self._step)
            self.logger.tbwriter.add_scalar(f"t{t}/loss/old_kd_with_temp", kd_with_temp_mean, self._step)

            ent_old = self.entropy(torch.cat(outputs[:t], dim=1))
            ent_old_model = self.entropy(torch.cat(outputs_old[:t], dim=1))
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

            if self.lambda_cos > 0:
                loss_dist_cos = nn.CosineEmbeddingLoss()(
                    embeddings, embeddings_old.detach(), torch.ones(targets.shape[0]).to(self.device)
                )
                loss_dist_cos = self.lambda_cos * loss_dist_cos
                self.logger.tbwriter.add_scalar(f"t{t}/loss_dist_cos", loss_dist_cos, self._step)
                loss += loss_dist_cos

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


# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out

        return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out
