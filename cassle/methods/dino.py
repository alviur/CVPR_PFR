import argparse
import distutils
from typing import Any, List, Sequence, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from cassle.losses.dino import DINOLoss
from cassle.methods.base import BaseMomentumModel
from cassle.utils.momentum import initialize_momentum_params
from cassle.utils.trunc_normal import trunc_normal_


class DINOHead(nn.Module):
    mlp: Any
    last_layer: Any

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int,
        use_bn: bool = True,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        """DINO head that takes as input the features of the encoder, projects them in a lower
        dimensional space and multiplies with the prototypes.

        Args:
            in_dim (int): number of dimensions of the input (aka encoder features).
            num_prototypes (int): number of prototypes.
            use_bn (bool, optional): whether to use batch norm in projector. Defaults to True.
            norm_last_layer (bool, optional): whether to l2-norm the last layer. Defaults to True.
            num_layers (int, optional): number of layers in projector. Defaults to 3.
            hidden_dim (int, optional): number of dimension in hidden layers. Defaults to 2048.
            bottleneck_dim (int, optional): number of dimensions in bottleneck. Defaults to 256.
        """

        super().__init__()

        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: List[Any] = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module):
        """Initializes weights with truncated normal and biases with zeros.

        Args:
            m (nn.Module): a layer of the DINO head.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the projector and the last layer (prototypes).

        Args:
            x (torch.Tensor): a batch of features.

        Returns:
            torch.Tensor: a batch of logits.
        """

        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINO(BaseMomentumModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        num_prototypes: int,
        norm_last_layer: bool,
        clip_grad: float,
        freeze_last_layer: bool,
        student_temperature: float,
        teacher_temperature: float,
        warmup_teacher_temperature: float,
        warmup_teacher_temperature_epochs: int,
        **kwargs,
    ):
        """Adds DINO head to the student and momentum DINO head to the teacher.

        Args:
            output_dim (int): number of prototypes.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            num_prototypes (int): number of prototypes.
            norm_last_layer (bool): whether or not to normalize the last layer (prototypes).
            clip_grad (float): threshold for gradient clipping.
            freeze_last_layer (bool): whether or not to freeze the last layer (prototypes).
            student_temperature (float): temperature for the student.
            teacher_temperature (float): temperature for the teacher.
            warmup_teacher_temperature (float): base temperature for the teacher.
            warmup_teacher_temperature_epochs (int): number of epochs of cosine annealing
                scheduling for teacher temperature.
        """

        super().__init__(**kwargs)

        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer

        # dino head
        self.head = DINOHead(
            in_dim=self.features_dim,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=output_dim,
            num_prototypes=num_prototypes,
            norm_last_layer=norm_last_layer,
        )

        # instantiate and initialize momentum dino head
        self.momentum_head = DINOHead(
            in_dim=self.features_dim,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=output_dim,
            num_prototypes=num_prototypes,
            norm_last_layer=norm_last_layer,
        )
        initialize_momentum_params(self.head, self.momentum_head)

        # dino loss
        self.dino_loss_func = DINOLoss(
            num_prototypes=num_prototypes,
            student_temp=student_temperature,
            warmup_teacher_temp=warmup_teacher_temperature,
            teacher_temp=teacher_temperature,
            warmup_teacher_temp_epochs=warmup_teacher_temperature_epochs,
            num_epochs=self.max_epochs,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(DINO, DINO).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("dino")

        # optimization settings
        parser.add_argument("--clip_grad", type=float, default=0)
        parser.add_argument("--freeze_last_layer", type=int, default=1)

        # dino head
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--num_prototypes", type=int, default=4096)
        parser.add_argument("--norm_last_layer", type=distutils.util.strtobool, default=True)

        # temperature settings
        parser.add_argument("--student_temperature", type=float, default=0.1)
        parser.add_argument("--teacher_temperature", default=0.07, type=float)
        parser.add_argument("--warmup_teacher_temperature", default=0.04, type=float)
        parser.add_argument("--warmup_teacher_temperature_epochs", default=50, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds DINO head parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.head.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (head, momentum_head) to the parent's momentum pairs.

        Returns:
            List[dict]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.head, self.momentum_head)]
        return super().momentum_pairs + extra_momentum_pairs

    def clip_gradients(self, clip: float):
        """Clips gradients after backward pass.

        Args:
            clip (float): threshold for gradient clipping.
        """

        for p in self.encoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)

    def on_train_epoch_start(self):
        """Updates the current epoch in DINO's loss object."""
        self.dino_loss_func.epoch = self.current_epoch

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the student (encoder and head).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().forward(X, *args, **kwargs)
        p = self.head(out["feats"])
        return {**out, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DINO reusing BaseMomentumModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where [X]
                is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DINO loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        # forward online encoder
        p1 = self.head(feats1)
        p2 = self.head(feats2)
        p = torch.cat((p1, p2))

        # forward momentum encoder
        p1_momentum = self.momentum_head(momentum_feats1)
        p2_momentum = self.momentum_head(momentum_feats2)
        p_momentum = torch.cat((p1_momentum, p2_momentum))

        # ------- contrastive loss -------
        dino_loss = self.dino_loss_func(p, p_momentum)

        self.log("dino_loss", dino_loss, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + dino_loss, "z": [p1, p2]})
        return out

    def on_after_backward(self):
        """Performs gradient clipping and zeros the gradients on the last layer (prototypes)."""

        # clip gradients
        if self.clip_grad:
            self.clip_gradients(self.clip_grad)
        # zero gradients on last layer
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None
