"""metrics for evaluation"""
"""如果要用做验证集指标，还需要将结果取平均"""
#! 目前这些指标计算只能用于验证集。按照公式计算，和实际软件跑出来的存在区别。
#! 这里只使用每个残基的单原子（比如openfold使用Cα，nucleotide一般用C3'或C4'）
#! 输出的是每个样本的分数
#todo openstructure貌似包含了我想要的关于蛋白质、核酸的原子的约束条件，可以留意一下
from typing import Optional
import torch


def lddt(
    ref_pos: torch.Tensor, 
    pred_pos: torch.Tensor, 
    mask: Optional[torch.Tensor] = None,
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    残基index距离阈值r为1
    ref_pos: [*, N, 3]
    pred_pos: [*, N, 3]
    mask: [*, N]
    return: [*]
    """
    if mask is None:
        mask = ref_pos.new_ones(ref_pos.shape[:-1])
    
    n = mask.shape[-1] # N
    # [*, N, N]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                ref_pos[..., None, :]
                - ref_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                pred_pos[..., None, :]
                - pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    # [*, N, N]
    dists_to_score = (
        (dmat_true < cutoff) 
        * mask[..., None] 
        * mask[..., None, :]
        * (1.0 - torch.eye(n, device=mask.device))
    )
    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25
    dims = (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score
