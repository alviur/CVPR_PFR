import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.utils.loss_and_miner_utils import convert_to_pairs


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class AlignUniformLoss(BaseMetricLossFunction):
    def __init__(
            self,
            avg_non_zero_only=True,
            align_w=1,
            unif_w=1,
            align_alpha=2.0,
            unif_t=2.0,
            **kwargs
    ):
        super().__init__()
        self.unif_t = unif_t
        self.align_alpha = align_alpha
        self.unif_w = unif_w
        self.align_w = align_w
        self.avg_non_zero_only = avg_non_zero_only
        # self.add_to_recordable_attributes(list_of_names=["num_non_zero_pos_pairs", "num_non_zero_neg_pairs"])

    def _align_loss(self, x, y):
        return (x - y).norm(p=2, dim=1).pow(self.align_alpha).mean()

    def compute_loss(self, embeddings, labels, indices_tuple):
        a1, p, a2, n = convert_to_pairs(indices_tuple, labels)
        if len(a1) > 1:
            x = embeddings[a1]
            y = embeddings[p]
            ul = uniform_loss(embeddings)
            assert not torch.isnan(ul)
            al = self._align_loss(x, y)
            assert not torch.isnan(al)
            loss = self.align_w * al + self.unif_w * ul
            # print(f"Loss: {loss} align: {al} uniform: {ul}")
            return {"loss": {"losses": loss, "indices": (a1, p), "reduction_type": "already_reduced"}}
        return {"loss": self.zero_loss()}
