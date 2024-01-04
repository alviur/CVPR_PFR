import torch
import torch.nn.functional as F
from typing import Optional


def simclr_distill_loss_func(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:

    device = z1.device

    b = z1.size(0)

    p = F.normalize(torch.cat([p1, p2]), dim=-1)
    z = F.normalize(torch.cat([z1, z2]), dim=-1)

    logits = torch.einsum("if, jf -> ij", p, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask.fill_diagonal_(True)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device)
    logit_mask.fill_diagonal_(True)
    logit_mask[:, b:].fill_diagonal_(True)
    logit_mask[b:, :].fill_diagonal_(True)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss


def simclr_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    extra_pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        temperature (float): temperature factor for the loss. Defaults to 0.1.
        extra_pos_mask (Optional[torch.Tensor]): boolean mask containing extra positives other
            than normal across-view positives. Defaults to None.

    Returns:
        torch.Tensor: SimCLR loss.
    """

    device = z1.device

    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)

    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)

    # if we have extra "positives"
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask, extra_pos_mask)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss


def manual_simclr_loss_func(
    z: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Manually computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): NxViewsxD Tensor containing projected features from the views.
        pos_mask (torch.Tensor): boolean mask containing all positives for z * z.T.
        neg_mask (torch.Tensor): boolean mask containing all negatives for z * z.T.
        temperature (float): temperature factor for the loss.

    Return:
        torch.Tensor: manual SimCLR loss.
    """

    z = F.normalize(z, dim=-1)

    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    negatives = torch.sum(torch.exp(logits) * neg_mask, dim=1, keepdim=True)
    exp_logits = torch.exp(logits)
    log_prob = torch.log(exp_logits / (exp_logits + negatives))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)

    indexes = pos_mask.sum(1) > 0
    pos_mask = pos_mask[indexes]
    mean_log_prob_pos = mean_log_prob_pos[indexes] / pos_mask.sum(1)

    # loss
    loss = -mean_log_prob_pos.mean()
    return loss
