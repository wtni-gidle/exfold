from typing import Dict, Mapping, List, Optional, Union, Tuple
import itertools
from functools import reduce, wraps
from operator import add
import numpy as np

import torch

from exfold.common import nucleic_constants as nc
from exfold.utils.geometry import Rigid3Array
from exfold.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    batched_gather,
)
from exfold.config import NUM_RES, NUM_MSA_SEQ

TensorDict = Dict[str, torch.Tensor]

def curry1(f):
    """Supply all arguments but the first."""
    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def cast_to_64bit_ints(protein: TensorDict) -> TensorDict:
    # We keep all ints as int64
    for k, v in protein.items():
        if v.dtype == torch.int32:
            protein[k] = v.type(torch.int64)

    return protein


def squeeze_features(protein: TensorDict) -> TensorDict:
    """
    Remove singleton and repeated dimensions in protein features.
    restype: [N_res, 4+1] -> [N_res]
    seq_length: [num_res] -> scalar
    """
    # Remove singleton dimensions in protein features.
    protein["restype"] = torch.argmax(protein["restype"], dim=-1)
    for k in [
        "msa",
        "seq_length",
        "residue_index",
    ]:
        if k in protein:
            final_dim = protein[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(protein[k]):
                    protein[k] = torch.squeeze(protein[k], dim=-1)
                else:
                    protein[k] = np.squeeze(protein[k], axis=-1)
    # Remove repeated values in protein features.
    for k in ["seq_length", "use_clamped_fape"]:
        if k in protein:
            protein[k] = protein[k][0]

    return protein


def make_seq_mask(protein: TensorDict) -> TensorDict:
    """
    seq_mask: [N_res] all ones
    """
    protein["seq_mask"] = torch.ones(
        protein["restype"].shape, dtype=torch.float32
    )
    return protein


def make_msa_mask(protein: TensorDict) -> TensorDict:
    """
    Mask features are all ones, but will later be zero-padded.
    msa_mask: [N_seq, N_res] all ones, N_seq=1 for no msa mode
    """
    protein["msa_mask"] = torch.ones(protein["msa"].shape, dtype=torch.float32)
    return protein


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
    

def make_backbone_atoms(protein: TensorDict) -> TensorDict:
    """
    Create backbone atom position and mask.
    Returns:
        glycos_N: [N_res, 3]
        glycos_N_mask: [N_res]
        C4_prime: [N_res, 3]
        C4_prime_mask: [N_res]
        P: [N_res, 3]
        P_mask: [N_res]
    """
    (
        protein["glycos_N"], 
        protein["glycos_N_mask"],
    ) = backbone_atom_fn(
        "glycos_N",
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )
    (
        protein["C4_prime"], 
        protein["C4_prime_mask"],
    ) = backbone_atom_fn(
        "C4'",
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )
    (
        protein["P"], 
        protein["P_mask"],
    ) = backbone_atom_fn(
        "P",
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )

    return protein


def get_backbone_frames(protein: TensorDict) -> TensorDict:
    """
    Compute backbone frames.
    Returns:
        backbone_rigid_tensor: [..., N_res, 4, 4]
        backbone_rigid_mask: [..., N_res]
    """
    # [..., N_res]
    restype = protein["restype"]
    batch_dims = len(restype.shape[:-1])
    # [..., N_res, 3]
    atom_P = protein["P"]
    C4_prime = protein["C4_prime"]
    glycos_N = protein["glycos_N"]
    # [..., N_res, rna_backbone_atom_num]
    all_atom_mask = protein["all_atom_mask"]

    # [..., ACGUX]
    restype_bb_mask = all_atom_mask.new_ones(
        (*restype.shape[:-1], nc.rna_restype_num + 1),
    )
    #* Assume that UNK residues do not have a backbone.
    restype_bb_mask[..., -1] = 0
    
    # [..., N_res]
    backbone_exists = batched_gather(
        restype_bb_mask,
        restype,
        dim=-1,
        no_batch_dims=batch_dims,
    )
    # [..., N_res]
    gt_backbone_exists = torch.min(all_atom_mask, dim=-1)[0] * backbone_exists

    #* The atom order here can not be changed.
    # [..., N_res]
    gt_backbone_frame = Rigid3Array.from_3_points_svd(
        p1=atom_P,  # P
        p2=C4_prime,  # C
        p3=glycos_N,  # N
    )
    # [..., N_res, 4, 4]
    gt_backbone_tensor = gt_backbone_frame.to_tensor4x4()
    
    protein["backbone_rigid_tensor"] = gt_backbone_tensor
    protein["backbone_rigid_mask"] = gt_backbone_exists

    return protein
    

def make_one_hot(
    x: torch.Tensor, 
    num_classes: int
) -> torch.Tensor:
    """
    return [..., num_classes]
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def shaped_categorical(
    probs: torch.Tensor, 
    epsilon: float = 1e-10
):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(
        torch.reshape(probs + epsilon, [-1, num_classes])
    )
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])


@curry1
def make_masked_msa(
    protein: TensorDict, 
    config, 
    replace_fraction: float, 
    seed: int
) -> TensorDict:
    """
    Create data for BERT on raw MSA.
    N_seq=1 for no msa mode; [MASK] is an additional token, not X or gap.
    Returns:
        bert_mask: [..., N_seq, N_res] 1: masked
        true_msa: [..., N_seq, N_res] gt msa
        msa: [..., N_seq, N_res] msa after mask
    """
    device = protein["msa"].device

    # Add a random amino acid uniformly.
    #* Since no MSA mode is currently used, gaps are not applied. [ACGUX]
    # [4+1] 
    random_aa = torch.tensor(
        [1 / nc.rna_restype_num] * nc.rna_restype_num + [0.0], 
        dtype=torch.float32, 
        device=device
    )
    # [..., N_seq, N_res, 4+1]
    categorical_probs = (
        config.uniform_prob * random_aa
        + config.same_prob * make_one_hot(
            protein["msa"], 
            nc.rna_restype_num + 1
        )
    )

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(
        reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))])
    )
    pad_shapes[1] = 1
    mask_prob = 1.0 - config.same_prob - config.uniform_prob
    assert mask_prob >= 0.0

    # Fill mask_prob in the [mask] location (4+1+1th column).
    # [..., N_seq, N_res, 4+1+1]
    categorical_probs = torch.nn.functional.pad(
        categorical_probs, pad_shapes, value=mask_prob,
    )

    # Each position in categorical_probs represents a probability value, 
    # indicating the probability of the original residue being replaced by current residue.
    g = None
    if seed is not None:
        g = torch.Generator(device=protein["msa"].device)
        g.manual_seed(seed)
    
    # mask_position: 1为被替换
    sample = torch.rand(protein["msa"].shape, device=device, generator=g)
    # [..., N_seq, N_res]
    mask_position = sample < replace_fraction
    # [..., N_seq, N_res]
    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, protein["msa"])

    # Mix real and masked MSA
    protein["bert_mask"] = mask_position.to(torch.float32)
    protein["true_msa"] = protein["msa"]
    protein["msa"] = bert_msa

    return protein


def make_msa_feat(protein: TensorDict) -> TensorDict:
    """
    Create and concatenate MSA features.
    msa_feat: [..., N_seq, N_res, 4+2] N_seq=1 for no msa mode
    target_feat: [..., N_res, 4+1]
    """
    # AUCGX
    restype_1hot = make_one_hot(protein["restype"], 4+1)
    target_feat = [restype_1hot]

    # AUCGX[mask] no gap for no msa mode
    msa_1hot = make_one_hot(protein["msa"], 4+2)
    msa_feat = [msa_1hot]

    protein["msa_feat"] = torch.cat(msa_feat, dim=-1)
    protein["target_feat"] = torch.cat(target_feat, dim=-1)

    return protein


@curry1
def select_feat(protein: TensorDict, feature_list: List) -> TensorDict:
    return {k: v for k, v in protein.items() if k in feature_list}


@curry1
def random_crop_to_size(
    protein: TensorDict,
    crop_size: int,
    shape_schema: Dict,
    seed: int = None,
) -> TensorDict:
    """
    Crop randomly to `crop_size`, or keep as is if shorter than that.
    Crop along the `NUM_RES` dimension.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=protein["seq_length"].device)
        g.manual_seed(seed)

    seq_length = protein["seq_length"]

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["seq_length"].device,
                generator=g,
        )[0])

    # 1st point of 1.2.8 Residue cropping 
    n = seq_length - num_res_crop_size
    if "use_clamped_fape" in protein and protein["use_clamped_fape"] == 1.:
        right_anchor = n
    else:
        x = _randint(0, n)
        right_anchor = n - x

    num_res_crop_start = _randint(0, right_anchor)

    for k, v in protein.items():
        if NUM_RES not in shape_schema[k]:
            continue

        slices = []
        for dim_size, dim in zip(shape_schema[k], v.shape):
            is_num_res = dim_size == NUM_RES
            crop_start = num_res_crop_start if is_num_res else 0
            crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices]

    protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)
    
    return protein


@curry1
def make_fixed_size(
    protein: TensorDict,
    shape_schema: Dict,
    num_res: int = 0,
) -> TensorDict:
    """
    Guess at the MSA and sequence dimension to make fixed size.
    Zero padding the `mask` ensures padded parts are excluded from attention. 
    `pair_mask` can be derived from `seq_mask`.
    """
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: 1, # for no msa mode
    }

    for k, v in protein.items():
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)

    return protein

