import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from cassle.losses.swav import swav_loss_func
from cassle.methods.base import BaseModel
from cassle.utils.sinkhorn_knopp import SinkhornKnopp


class SwAV(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        num_prototypes: int,
        sk_iters: int,
        sk_epsilon: float,
        temperature: float,
        queue_size: int,
        epoch_queue_starts: int,
        freeze_prototypes_epochs: int,
        **kwargs,
    ):
        """Implements SwAV (https://arxiv.org/abs/2006.09882).

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            num_prototypes (int): number of prototypes.
            sk_iters (int): number of iterations for the sinkhorn-knopp algorithm.
            sk_epsilon (float): weight for the entropy regularization term.
            temperature (float): temperature for the softmax normalization.
            queue_size (int): number of samples to hold in the queue.
            epoch_queue_starts (int): epochs the queue starts.
            freeze_prototypes_epochs (int): number of epochs during which the prototypes are frozen.
        """

        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.sk_iters = sk_iters
        self.sk_epsilon = sk_epsilon
        self.temperature = temperature
        self.queue_size = queue_size
        self.epoch_queue_starts = epoch_queue_starts
        self.freeze_prototypes_epochs = freeze_prototypes_epochs

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # prototypes
        self.prototypes = nn.utils.weight_norm(nn.Linear(output_dim, num_prototypes, bias=False))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SwAV, SwAV).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("swav")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # queue settings
        parser.add_argument("--queue_size", default=3840, type=int)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--num_prototypes", type=int, default=3000)
        parser.add_argument("--sk_epsilon", type=float, default=0.05)
        parser.add_argument("--sk_iters", type=int, default=3)
        parser.add_argument("--freeze_prototypes_epochs", type=int, default=1)
        parser.add_argument("--epoch_queue_starts", type=int, default=15)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and prototypes parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.prototypes.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def on_train_start(self):
        """Gets the world size and sets it in the sinkhorn and the queue."""

        super().on_train_start()

        # sinkhorn-knopp needs the world size
        world_size = self.trainer.world_size if self.trainer else 1
        self.sk = SinkhornKnopp(self.sk_iters, self.sk_epsilon, world_size)
        # queue also needs the world size
        if self.queue_size > 0:
            self.register_buffer(
                "queue",
                torch.zeros(
                    2,
                    self.queue_size // world_size,
                    self.output_dim,
                    device=self.device,
                ),
            )

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder, the projector and the prototypes.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent,
                the projected features and the logits.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        z = F.normalize(z)
        p = self.prototypes(z)
        return {**out, "z": z, "p": p}

    @torch.no_grad()
    def get_assignments(self, preds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Computes cluster assignments from logits, optionally using a queue.

        Args:
            preds (List[torch.Tensor]): a batch of logits.

        Returns:
            List[torch.Tensor]: assignments for each sample in the batch.
        """

        bs = preds[0].size(0)
        assignments = []
        for i, p in enumerate(preds):
            # optionally use the queue
            if self.queue_size > 0 and self.current_epoch >= self.epoch_queue_starts:
                p_queue = self.prototypes(self.queue[i])  # type: ignore
                p = torch.cat((p, p_queue))
            # compute assignments with sinkhorn-knopp
            assignments.append(self.sk(p)[:bs])
        return assignments

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SwAV reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SwAV loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)
        z1_norm = F.normalize(z1)
        z2_norm = F.normalize(z2)

        p1 = self.prototypes(z1_norm)
        p2 = self.prototypes(z2_norm)

        # ------- swav loss -------
        preds = [p1, p2]
        assignments = self.get_assignments(preds)
        swav_loss = swav_loss_func(preds, assignments, self.temperature)

        # ------- update queue -------
        if self.queue_size > 0:
            z = torch.stack((z1_norm, z2_norm))
            self.queue[:, z.size(1) :] = self.queue[:, : -z.size(1)].clone()
            self.queue[:, : z.size(1)] = z.detach()

        self.log("train_swav_loss", swav_loss, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + swav_loss, "z": [z1, z2], "p": [p1, p2]})
        return out

    def on_after_backward(self):
        """Zeroes the gradients of the prototypes."""
        if self.current_epoch < self.freeze_prototypes_epochs:
            for p in self.prototypes.parameters():
                p.grad = None
