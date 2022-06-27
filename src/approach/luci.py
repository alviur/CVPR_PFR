import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader

from .learning_approach import Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Learning_Appr):
    """
    Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in:
    http://dahua.me/publications/dhl19_increclass.pdf
    """

    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, lamda_base=5, lamda_mr=1., dist=.5, K=2, orig_scheduler=False,
                 remove_less_forget=False,
                 remove_margin_ranking=False, remove_adapt_lamda=False, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)

        self.lamda_base = lamda_base
        self.lamda_mr = lamda_mr
        self.dist = dist
        self.K = K
        self.orig_scheduler = orig_scheduler
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamda_base
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Sec. 4.1: "we used the method proposed in [29] based on herd selection"
        # Sec. 4.1: "first one stores a constant number of samples for each old class (e.g. R_per=20) (...) we adopt
        # the first strategy"
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamda_base', default=5., type=float, required=False,
                            help='Lambda for distillation loss (default=%(default)s)')
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamda_mr', default=1., type=float, required=False,
                            help='Lambda for the MR loss(default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')
        parser.add_argument('--orig_scheduler', action='store_true', required=False,
                            help='Activate the LR scheduling from the original experiments (default=%(default)s)')
        # Flags for ablating the approach
        parser.add_argument('--remove_less_forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove_margin_ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove_adapt_lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        # Don't update heads when Less-Forgetting constraint is activated (from original code)
        if self.less_forget:
            parameters = [{'params': self.model.model.parameters()}, {'params': self.model.heads[-1].parameters()}]
        else:
            parameters = self.model.parameters()

        optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

        # Sec 4.1: "For CIFAR100, the learning rate starts from 0.1 and is divided by 10 after 80 and 120 epochs
        # (160 epochs in total).
        # For ImageNet, the learning rates also starts from 0.1 and is divided by 10 every 30 epochs
        # (90 epochs in total)
        if self.orig_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], gamma=0.1)
        return optimizer

    def pre_train_process(self, t, trn_loader):
        # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
        # negative values"
        if t == 0:
            "network is not detected as ResNet and last ReLU needs to be removed!"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                            old_block.conv2, old_block.bn2, old_block.downsample)
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        self.model.to(self.device)

        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma

            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True

            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamda_base * math.sqrt(sum([h.out_features for h in self.model.heads[:-1]])
                                                         / self.model.heads[-1].out_features)

        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, I think that is not mentioned in the paper and doesn't seem to make a significance difference.
        # TODO: Add imprint weights (from original code)

        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        # Create a joint dataset with the exemplars
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        
        super().train_loop(t, trn_loader, val_loader)

        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        # Sec 3.5: "we find that the so-called class balance finetune can improve the performance moderately
        # in practice"
        # Sec 4.4: "CBF has a relatively small effect on this dataset"
        # I haven't found the class balance finetune in the original code
        # TODO: implement Class Balance Finetune

        # From LwF
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

    # Runs a single epoch
    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        total_loss = 0.0
        total_examples = 0
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            # Forward current model
            outputs, features = self.model(images, return_features=True)

            # Forward previous model
            ref_outputs = None
            ref_features = None
            if t > 0:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)

            loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_examples += len(targets)

        # Make step on the LR scheduler (as in the original paper's setting)
        if self.orig_scheduler:
            self.scheduler.step()
        
        return total_loss / total_examples

    # Returns the loss value
    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None):
        if ref_outputs is None or ref_features is None:
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).to(self.device))
                    loss_mr *= self.lamda_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            # Eq. 9: integrated objective
            loss = loss_dist + loss_ce + loss_mr
        return loss


# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
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

        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
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
