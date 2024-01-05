import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper


def pfr_distill_wrapper(Method=object):
    class PfrDistillWrapper(base_distill_wrapper(Method)):
        def __init__(
            self,
            distill_lamb: float,
            distill_proj_hidden_dim: int,
            distill_barlow_lamb: float,
            distill_scale_loss: float,
            **kwargs
        ):
            super().__init__(**kwargs)

            output_dim = kwargs["output_dim"]
            self.distill_lamb = distill_lamb
            self.distill_barlow_lamb = distill_barlow_lamb
            self.distill_scale_loss = distill_scale_loss

            self.distill_predictor = nn.Sequential(
                nn.Linear(self.features_dim, int(self.features_dim/2)),
                nn.BatchNorm1d(int(self.features_dim/2)),
                nn.ReLU(),
                nn.Linear(int(self.features_dim/2), self.features_dim),
            )

            # PFR criterion
            self.criterion = nn.CosineSimilarity(dim=1)

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            parser.add_argument("--distill_barlow_lamb", type=float, default=5e-3)
            parser.add_argument("--distill_scale_loss", type=float, default=0.1)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.distill_predictor.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            
            f1, f2 = out["feats"]

            frozen_f1, frozen_f2 = out["frozen_feats"]

            p1 = self.distill_predictor(f1)
            p2 = self.distill_predictor(f2)

            distill_loss = (-(self.criterion(p1, frozen_f1.detach()).mean()
                                        + self.criterion(p2, frozen_f2.detach()).mean()) * 0.5)

            self.log(
                "train_decorrelative_distill_loss", distill_loss, on_epoch=True, sync_dist=True
            )

            return out["loss"] + self.distill_lamb * distill_loss

    return PfrDistillWrapper
