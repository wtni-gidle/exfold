from typing import Dict, Optional, Tuple, Union
import ml_collections
import torch
import torch.nn as nn
import logging

from exfold.utils.feats import backbone_atom_fn
from exfold.utils.geometry import (
    Vec3Array, 
    Rigid3Array, 
    dihedral_angle, 
    square_euclidean_distance
)


def softmax_cross_entropy(
    logits: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        logits: 
            [*, no_bins] does not represent probability values, but linear layer outputs.
        labels: 
            [*, no_bins]
    Returns:
        loss: 
            [*]
    """
    loss = -1 * torch.sum(
        labels * nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def compute_distogram_loss(
    logits: torch.Tensor,
    atom_position: torch.Tensor,
    atom_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    [0, min_bin]: 1
    (min_bin, max_bin]: no_bins - 2
    (max_bin, +infinity]: 1
    Args:
        logits: [*, N_res, N_res, no_bins]
        atom_position: [*, N_res, 3]
        atom_mask: [*, N_res]

    Return: 
        loss: scalar
    """
    # [no_bins-1,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    # [*, N_res, N_res, 1] distance square
    dists = torch.sum(
        (atom_position[..., None, :] - atom_position[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )
    # [*, N_res, N_res] 值为 0 - (no_bins-1)
    true_bins = torch.sum(dists > boundaries, dim=-1)
    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins), # [*, N_res, N_res, no_bins]
    )
    # [*, N_res, N_res]
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    loss = errors * square_mask
    # [*, N_res]
    loss = torch.sum(loss, dim=-1)
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    # [*]
    loss = torch.sum(loss, dim=-1)

    # Average over the batch dimensions
    # scalar
    loss = torch.mean(loss)

    return loss


def distogram_loss(
    logits: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    atom_name: str,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    Compute distogram loss given atom name.
    """
    atom_position, atom_mask = backbone_atom_fn(
        atom_name=atom_name,
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )

    return compute_distogram_loss(
        logits=logits,
        atom_position=atom_position,
        atom_mask=atom_mask,
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        eps=eps,
    )


def masked_msa_loss(
    logits: torch.Tensor, 
    true_msa: torch.Tensor, 
    bert_mask: torch.Tensor, 
    num_classes: int, 
    eps: float = 1e-8, 
    **kwargs
) -> torch.Tensor:
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: 
            [*, N_seq, N_res, 4+1+1] predicted residue distribution
        true_msa: 
            [*, N_seq, N_res] true MSA
        bert_mask: 
            [*, N_seq, N_res] MSA mask, 1: masked
    Returns:
        Masked MSA loss: scalar
    """
    # [*, N_seq, N_res]
    errors = softmax_cross_entropy(
        logits, 
        torch.nn.functional.one_hot(true_msa, num_classes=num_classes)
    )

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    # [*, N_seq]
    loss = torch.sum(loss, dim=-1)
    scale = 0.5 # `scale` might be used to ensure stability during FP16 training.
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    # [*]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    # Average over the batch dimensions
    # scalar
    loss = torch.mean(loss)

    return loss


# region: fape
def compute_fape(
    pred_frames: Rigid3Array,
    target_frames: Rigid3Array,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    # x_ij [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.inverse()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    # x_ij true [*, N_frames, N_pts, 3]
    local_target_pos = target_frames.inverse()[..., None].apply(
        target_positions[..., None, :, :],
    )
    # [*, N_frames, N_pts]
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    # Since frames_mask=1 iff positions_mask=1 for all three atoms,
    # frames_mask may be sufficient.
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


#todo Currently we compute fape loss for centroid of three backbone atoms.
#todo Need to perform an experiment and cancel this comment.
def intermediate_fape_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    """
    Compute fape loss in every iteration/layer.

    Args:
        backbone_rigid_tensor : [*, N_res, 4, 4]
        backbone_rigid_mask : [*, N_res]
        traj : [no_blocks, *, N_res, 4, 4]
        use_clamped_fape: [*]

    Returns
    -------
    [*]
    """
    assert traj.shape[-1] == 4
    # [no_blocks, *, N_res]
    pred_frame = Rigid3Array.from_tensor4x4(traj)
    # [no_blocks, *, N_res]
    gt_frame = Rigid3Array.from_tensor4x4(backbone_rigid_tensor).unsqueeze(0)

    # The original code in openfold seems to be strange,
    # We made some changes about `use_clamped_fape`.

    unclamped_fape_loss = compute_fape(
        pred_frame,
        gt_frame,
        backbone_rigid_mask[None],
        pred_frame.translation.to_tensor(),
        gt_frame.translation.to_tensor(),
        backbone_rigid_mask[None],
        length_scale=loss_unit_distance,
        l1_clamp_distance=None,
        eps=eps,
    )
    # `use_clamped_fape` is not None for train and eval mode.
    if use_clamped_fape is not None:
        clamped_fape_loss = compute_fape(
            pred_frame,
            gt_frame,
            backbone_rigid_mask[None],
            pred_frame.translation.to_tensor(),
            gt_frame.translation.to_tensor(),
            backbone_rigid_mask[None],
            length_scale=loss_unit_distance,
            l1_clamp_distance=clamp_distance,
            eps=eps,
        )
        
        fape_loss = clamped_fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    else:
        fape_loss = unclamped_fape_loss

    return fape_loss


def final_fape_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
):
    """
    Compute fape loss for the last iteration/layer.
    
    Args:
        backbone_rigid_tensor : [*, N_res, 4, 4]
        backbone_rigid_mask : [*, N_res]
        all_atom_positions: [*, N_res, rna_backbone_atom_num, 3]
        all_atom_mask: [*, N_res, rna_backbone_atom_num]
        pred_positions: [*, N_res, rna_backbone_atom_num, 3]

    Returns:
        [*]
    """
    assert traj.shape[-1] == 4
    # [*, N_res]
    pred_frame = Rigid3Array.from_tensor4x4(traj)
    # [*, N_res]
    gt_frame = Rigid3Array.from_tensor4x4(backbone_rigid_tensor)
    # [*]
    batch_dims = gt_frame.shape[:-1]
    # [*, N_res * rna_backbone_atom_num, 3]
    pred_positions = pred_positions.view(*batch_dims, -1, 3)
    # [*, N_res * rna_backbone_atom_num, 3]
    gt_positions = all_atom_positions.view(*batch_dims, -1, 3)
    # [*, N_res * rna_backbone_atom_num]
    positions_mask = all_atom_mask.view(*batch_dims, -1)

    fape = compute_fape(
        pred_frame,
        gt_frame,
        backbone_rigid_mask,
        pred_positions,
        gt_positions,
        positions_mask,
        length_scale=loss_unit_distance,
        l1_clamp_distance=clamp_distance, 
        eps=eps,
    )

    return fape


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    intermediate_loss = intermediate_fape_loss(
        traj=out["sm"]["frames"],
        **{**batch, **config.intermediate},
    )
    final_loss = final_fape_loss(
        traj=out["sm"]["frames"][-1],
        pred_positions=out["sm"]["positions"][-1],
        **{**batch, **config.final},
    )
    
    loss = intermediate_loss * config.intermediate.weight + \
        final_loss * config.final.weight

    # Average over the batch dimension
    loss = torch.mean(loss)
    
    return loss
# endregion
    

def PCCP_dihedral_loss(
    logits: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    max_dist: float,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    no_bins = 36 + 1, 最后一维为no contact
    Args:
        logits: 
            [*, N_res, N_res, no_bins]
        all_atom_positions: 
            [*, N_res, rna_backbone_atom_num, 3]
        all_atom_mask:
            [*, N_res, rna_backbone_atom_num]
    
    Returns:
        loss: scalar
    """
    # [no_bins,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=logits.device,
    )
    # Unlike distogram loss, we need to adjust for the boundaries of angles 
    # to ensure that the gt angles will necessarily fall within one of the intervals.
    boundaries[0] -= 1e-5
    boundaries[-1] += 1e-5
    
    atom_P, atom_P_mask = backbone_atom_fn(
        atom_name="P",
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )
    C4_prime, C4_prime_mask = backbone_atom_fn(
        atom_name="C4'",
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )
    # [*, N_res]
    # atom_P = Vec3Array.from_array(atom_P.to(dtype=torch.float32))
    # C4_prime = Vec3Array.from_array(C4_prime.to(dtype=torch.float32))
    #TODO need to check whether outputs out of forward() is in fp32
    atom_P = Vec3Array.from_array(atom_P)
    C4_prime = Vec3Array.from_array(C4_prime)
    
    # [*, N_res, N_res, 1]
    dihedrals = dihedral_angle(
        atom_P.unsqueeze(-1),
        C4_prime.unsqueeze(-1),
        C4_prime.unsqueeze(-2),
        atom_P.unsqueeze(-2),
        degree=True
    )[..., None]
    # [*, N_res, N_res] 值为 1 - (no_bins-1)
    true_bins = torch.sum(dihedrals > boundaries, dim=-1)
    # [*, N_res, N_res] 值为 0 - (no_bins-2)
    true_bins -= 1

    # [*, N_res, N_res]
    C_dist = square_euclidean_distance(
        C4_prime.unsqueeze(-1),
        C4_prime.unsqueeze(-2),
        eps=0.
    )
    # [*, N_res, N_res]
    contact = C_dist > (max_dist ** 2)
    true_bins[contact] = no_bins - 1

    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    atom_mask = atom_P_mask * C4_prime_mask
    # [*, N_res, N_res]
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    loss = errors * square_mask
    # [*, N_res]
    loss = torch.sum(loss, dim=-1)
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    # [*]
    loss = torch.sum(loss, dim=-1)

    # Average over the batch dimensions
    # scalar
    loss = torch.mean(loss)

    return loss


def PNNP_dihedral_loss(
    logits: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    max_dist: float,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    no_bins = 36 + 1, 最后一维为no contact
    Args:
        logits: 
            [*, N_res, N_res, no_bins]
        all_atom_positions: 
            [*, N_res, rna_backbone_atom_num, 3]
        all_atom_mask:
            [*, N_res, rna_backbone_atom_num]
    
    Returns:
        loss: scalar
    """
    # [no_bins,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=logits.device,
    )
    # Unlike distogram loss, we need to adjust for the boundaries of angles 
    # to ensure that the gt angles will necessarily fall within one of the intervals.
    boundaries[0] -= 1e-5
    boundaries[-1] += 1e-5
    
    atom_P, atom_P_mask = backbone_atom_fn(
        atom_name="P",
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )
    glycos_N, glycos_N_mask = backbone_atom_fn(
        atom_name="glycos_N",
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )
    # [*, N_res]
    # atom_P = Vec3Array.from_array(atom_P.to(dtype=torch.float32))
    # C4_prime = Vec3Array.from_array(C4_prime.to(dtype=torch.float32))
    #TODO need to check whether outputs out of forward() is in fp32
    atom_P = Vec3Array.from_array(atom_P)
    glycos_N = Vec3Array.from_array(glycos_N)
    
    # [*, N_res, N_res, 1]
    dihedrals = dihedral_angle(
        atom_P.unsqueeze(-1),
        glycos_N.unsqueeze(-1),
        glycos_N.unsqueeze(-2),
        atom_P.unsqueeze(-2),
        degree=True
    )[..., None]
    # [*, N_res, N_res] 值为 1 - (no_bins-1)
    true_bins = torch.sum(dihedrals > boundaries, dim=-1)
    # [*, N_res, N_res] 值为 0 - (no_bins-2)
    true_bins -= 1

    # [*, N_res, N_res]
    N_dist = square_euclidean_distance(
        glycos_N.unsqueeze(-1),
        glycos_N.unsqueeze(-2),
        eps=0.
    )
    # [*, N_res, N_res]
    contact = N_dist > (max_dist ** 2)
    true_bins[contact] = no_bins - 1

    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    atom_mask = atom_P_mask * glycos_N_mask
    # [*, N_res, N_res]
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    loss = errors * square_mask
    # [*, N_res]
    loss = torch.sum(loss, dim=-1)
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    # [*]
    loss = torch.sum(loss, dim=-1)

    # Average over the batch dimensions
    # scalar
    loss = torch.mean(loss)

    return loss


def CNNC_dihedral_loss(
    logits: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    max_dist: float,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    no_bins = 36 + 1, 最后一维为no contact
    Args:
        logits: 
            [*, N_res, N_res, no_bins]
        all_atom_positions: 
            [*, N_res, rna_backbone_atom_num, 3]
        all_atom_mask:
            [*, N_res, rna_backbone_atom_num]
    
    Returns:
        loss: scalar
    """
    # [no_bins,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=logits.device,
    )
    # Unlike distogram loss, we need to adjust for the boundaries of angles 
    # to ensure that the gt angles will necessarily fall within one of the intervals.
    boundaries[0] -= 1e-5
    boundaries[-1] += 1e-5
    
    C4_prime, C4_prime_mask = backbone_atom_fn(
        atom_name="C4'",
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )
    glycos_N, glycos_N_mask = backbone_atom_fn(
        atom_name="glycos_N",
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask
    )
    # [*, N_res]
    # atom_P = Vec3Array.from_array(atom_P.to(dtype=torch.float32))
    # C4_prime = Vec3Array.from_array(C4_prime.to(dtype=torch.float32))
    #TODO need to check whether outputs out of forward() is in fp32
    C4_prime = Vec3Array.from_array(C4_prime)
    glycos_N = Vec3Array.from_array(glycos_N)
    
    # [*, N_res, N_res, 1]
    dihedrals = dihedral_angle(
        C4_prime.unsqueeze(-1),
        glycos_N.unsqueeze(-1),
        glycos_N.unsqueeze(-2),
        C4_prime.unsqueeze(-2),
        degree=True
    )[..., None]
    # [*, N_res, N_res] 值为 1 - (no_bins-1)
    true_bins = torch.sum(dihedrals > boundaries, dim=-1)
    # [*, N_res, N_res] 值为 0 - (no_bins-2)
    true_bins -= 1

    # [*, N_res, N_res]
    N_dist = square_euclidean_distance(
        glycos_N.unsqueeze(-1),
        glycos_N.unsqueeze(-2),
        eps=0.
    )
    # [*, N_res, N_res]
    contact = N_dist > (max_dist ** 2)
    true_bins[contact] = no_bins - 1

    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    atom_mask = C4_prime_mask * glycos_N_mask
    # [*, N_res, N_res]
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    loss = errors * square_mask
    # [*, N_res]
    loss = torch.sum(loss, dim=-1)
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    # [*]
    loss = torch.sum(loss, dim=-1)

    # Average over the batch dimensions
    # scalar
    loss = torch.mean(loss)

    return loss


class End2EndLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def loss(
        self, 
        out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """
        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                atom_name="glycos_N"
                **{**batch, **self.config.distogram},
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
        }

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        # AlphaFold2: To decrease the relative importance of short sequences, 
        # we multiply the final loss of each training example by the square root
        # of the number of residues after cropping. This implies equal weighting 
        # for all proteins that are longer than the crop size, 
        # and a square-root penalty for the shorter ones.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["restype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
    def forward(
        self, 
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses


class GeometryLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def loss(
        self,
        out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """
        loss_fns = {
            "PP": lambda: distogram_loss(
                logits=out["PP_logits"],
                atom_name="P",
                **{**batch, **self.config.PP},
            ),
            "CC": lambda: distogram_loss(
                logits=out["CC_logits"],
                atom_name="C4'",
                **{**batch, **self.config.CC},
            ),
            "NN": lambda: distogram_loss(
                logits=out["NN_logits"],
                atom_name="glycos_N",
                **{**batch, **self.config.NN},
            ),
            "PCCP": lambda: PCCP_dihedral_loss(
                logits=out["PCCP_logits"],
                **{**batch, **self.config.PCCP},
            ),
            "PNNP": lambda: PNNP_dihedral_loss(
                logits=out["PNNP_logits"],
                **{**batch, **self.config.PNNP},
            ),
            "CNNC": lambda: CNNC_dihedral_loss(
                logits=out["CNNC_logits"],
                **{**batch, **self.config.CNNC},
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
        }

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        # AlphaFold2: To decrease the relative importance of short sequences, 
        # we multiply the final loss of each training example by the square root
        # of the number of residues after cropping. This implies equal weighting 
        # for all proteins that are longer than the crop size, 
        # and a square-root penalty for the shorter ones.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["restype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
    def forward(
        self, 
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses
