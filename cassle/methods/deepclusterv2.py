import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from cassle.losses.deepclusterv2 import deepclusterv2_loss_func
from cassle.methods.base import BaseModel
from cassle.utils.kmeans import KMeans


class DeepClusterV2(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        num_prototypes: Sequence[int],
        temperature: float,
        kmeans_iters: int,
        **kwargs,
    ):
        """Implements DeepCluster V2 (https://arxiv.org/abs/2006.09882).

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            num_prototypes (Sequence[int]): number of prototypes.
            temperature (float): temperature for the softmax.
            kmeans_iters (int): number of iterations for k-means clustering.
        """

        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.temperature = temperature
        self.num_prototypes = num_prototypes
        self.kmeans_iters = kmeans_iters

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # prototypes
        self.prototypes = nn.ModuleList(
            [nn.Linear(output_dim, np, bias=False) for np in num_prototypes]
        )
        # normalize and set requires grad to false
        for proto in self.prototypes:
            for params in proto.parameters():
                params.requires_grad = False
            proto.weight.copy_(F.normalize(proto.weight.data.clone(), dim=-1))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(DeepClusterV2, DeepClusterV2).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("deepclusterv2")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--num_prototypes", type=int, nargs="+", default=[3000, 3000, 3000])
        parser.add_argument("--kmeans_iters", type=int, default=10)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and prototypes parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def on_train_start(self):
        """Gets the world size and initializes the memory banks."""

        super().on_train_start()

        #  k-means needs the world size and the dataset size
        self.world_size = self.trainer.world_size if self.trainer else 1
        self.dataset_size = getattr(self, "dali_epoch_size", None) or len(
            self.trainer.train_dataloader.dataset
        )

        # build k-means helper object
        self.kmeans = KMeans(
            world_size=self.world_size,
            rank=self.global_rank,
            num_crops=self.num_crops,
            dataset_size=self.dataset_size,
            proj_features_dim=self.output_dim,
            num_prototypes=self.num_prototypes,
            kmeans_iters=self.kmeans_iters,
        )

        # initialize memory banks
        size_memory_per_process = len(self.trainer.train_dataloader) * self.batch_size
        self.register_buffer(
            "local_memory_index",
            torch.zeros(size_memory_per_process).long().to(self.device, non_blocking=True),
        )
        self.register_buffer(
            "local_memory_embeddings",
            F.normalize(
                torch.randn(self.num_crops, size_memory_per_process, self.output_dim), dim=-1
            ).to(self.device, non_blocking=True),
        )

    def on_train_epoch_start(self) -> None:
        """Prepares assignments and prototype centroids for the next epoch."""

        if self.current_epoch == 0:
            self.assignments = -torch.ones(
                len(self.num_prototypes), self.dataset_size, device=self.device
            ).long()
        else:
            self.assignments, centroids = self.kmeans.cluster_memory(
                self.local_memory_index, self.local_memory_embeddings
            )
            for proto, centro in zip(self.prototypes, centroids):
                proto.weight.copy_(centro)

    def update_memory_banks(self, idxs: torch.Tensor, z: torch.Tensor, batch_idx: int) -> None:
        """Updates DeepClusterV2's memory banks of indices and features.

        Args:
            idxs (torch.Tensor): set of indices of the samples of the current batch.
            z (torch.Tensor): projected features of the samples of the current batch.
            batch_idx (int): batch index relative to the current epoch.
        """

        start_idx, end_idx = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
        self.local_memory_index[start_idx:end_idx] = idxs
        for c, z_c in enumerate(z):
            self.local_memory_embeddings[c][start_idx:end_idx] = z_c.detach()

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
        z = F.normalize(self.projector(out["feats"]))
        p = torch.stack([p(z) for p in self.prototypes])
        return {**out, "z": z, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DeepClusterV2 reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DeepClusterV2 loss and classification loss.
        """

        idxs = batch[f"task{self.current_task_idx}"][0].squeeze(-1)

        out = super().training_step(batch, batch_idx)
        feats1, feats2 = out["feats"]

        z1 = F.normalize(self.projector(feats1))
        z2 = F.normalize(self.projector(feats2))

        p1 = torch.stack([proto(z1) for proto in self.prototypes])
        p2 = torch.stack([proto(z2) for proto in self.prototypes])

        # ------- deepclusterv2 loss -------
        preds = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
        assignments = self.assignments[:, idxs]
        deepcluster_loss = deepclusterv2_loss_func(preds, assignments, self.temperature)

        # ------- update memory banks -------
        self.update_memory_banks(idxs, [z1, z2], batch_idx)

        self.log("train_deepcluster_loss", deepcluster_loss, on_epoch=True, sync_dist=True)

        out.update({"loss": out["loss"] + deepcluster_loss, "z": [z1, z2], "p": [p1, p2]})
        return out
