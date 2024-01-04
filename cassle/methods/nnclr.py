import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from cassle.losses.nnclr import nnclr_loss_func
from cassle.methods.base import BaseModel
from cassle.utils.gather_layer import gather


class NNCLR(BaseModel):
    queue: torch.Tensor

    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        temperature: float,
        queue_size: int,
        **kwargs,
    ):
        """Implements NNCLR (https://arxiv.org/abs/2104.14548).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons in the hidden layers of the predictor.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """
        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # queue
        self.register_buffer("queue", torch.randn(self.queue_size, output_dim))
        self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(NNCLR, NNCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("nnclr")

        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=4096)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.2)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor, y: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
        """

        z = gather(z)
        y = gather(y)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        self.queue_y[ptr : ptr + batch_size] = y  # type: ignore
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    @torch.no_grad()
    def find_nn(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbor of a sample.

        Args:
            z (torch.Tensor): a batch of projected features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """

        idx = (z @ self.queue.T).max(dim=1)[1]
        nn = self.queue[idx]
        return idx, nn

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for NNCLR reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y]
                where [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of NNCLR loss and classification loss.
        """

        targets = batch[f"task{self.current_task_idx}"][-1]

        out = super().training_step(batch, batch_idx)
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # find nn
        idx1, nn1 = self.find_nn(z1)
        _, nn2 = self.find_nn(z2)

        # ------- contrastive loss -------
        nnclr_loss = (
            nnclr_loss_func(nn1, p2, temperature=self.temperature) / 2
            + nnclr_loss_func(nn2, p1, temperature=self.temperature) / 2
        )

        # compute nn accuracy
        b = targets.size(0)
        nn_acc = (targets == self.queue_y[idx1]).sum() / b

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1, targets)

        metrics = {
            "train_nnclr_loss": nnclr_loss,
            "train_nn_acc": nn_acc,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + nnclr_loss, "z1": [z1, z2]})
        return out
