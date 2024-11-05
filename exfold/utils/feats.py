from typing import Optional, Union, Tuple

import torch

from exfold.utils.geometry import Rigid3Array

from exfold.common import nucleic_constants as nc


def backbone_atom_fn(
    atom_name: str,
    all_atom_positions: torch.Tensor, 
    all_atom_mask: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    supported atom_name: glycos_N, C4', P
    Returns:
        position: [..., N_res, 3]
        mask: [..., N_res]
    """
    if atom_name == "glycos_N":
        atom_name = "N9"
    assert atom_name in nc.rna_backbone_atoms["A"], f"unsupported atom name: {atom_name}"

    atom_idx = nc.rna_backbone_atoms["A"].index(atom_name)
    # [..., N_res, rna_backbone_atom_num, 3] -> [..., N_res,3]
    position = all_atom_positions[..., atom_idx, :]

    if all_atom_mask is not None:
        # [..., N_res, rna_backbone_atom_num] -> [..., N_res]
        mask = all_atom_mask[..., atom_idx]
        return position, mask
    else:
        return position


def backbone_and_literature_positions_to_atom3_pos(
    backbone_to_global: Rigid3Array,
    restype: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Apply backbone_to_global frame to literature positions in bakbone.
    Args:
        backbone_to_global: [*, N_res] Rigid3Array
        restype: [*, N_res]
        lit_positions: [5(ACGUX), rna_backbone_atom_num, 3]

    Returns:
        [*, N_res, rna_backbone_atom_num, 3]
    """
    # [*, N_res, rna_backbone_atom_num, 3]
    lit_positions = lit_positions[restype]
    # [*, N_res, 1]
    backbone_to_global = backbone_to_global.unsqueeze(-1)
    # [*, N_res, 3, 3]
    pred_positions = backbone_to_global.apply(lit_positions)

    return pred_positions
