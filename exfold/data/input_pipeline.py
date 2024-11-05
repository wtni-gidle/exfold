from typing import List, Callable, Dict
import random
import torch

from exfold.data import data_transforms

TensorDict = Dict[str, torch.Tensor]

def nonensembled_transform_fns(mode_cfg) -> List[Callable]:
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.squeeze_features,
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
    ]

    if mode_cfg.supervised:
        transforms.extend(
            [
                # data_transforms.make_backbone_atoms,
                # 或许应该再来一个二面角的gt
                data_transforms.get_backbone_frames,
            ]
        )

    return transforms


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []
    if "masked_msa" in common_cfg:
        transforms.append(
            # 每次iteration的seed不同, 因此每次的mask都是不同的
            data_transforms.make_masked_msa(
                common_cfg.masked_msa, 
                mode_cfg.masked_msa_replace_fraction,
                seed=None,
            )
        )

    transforms.append(data_transforms.make_msa_feat)
    crop_feats = dict(common_cfg.feat)

    if mode_cfg.fixed_size:
        transforms.append(data_transforms.select_feat(list(crop_feats)))
        transforms.append(
            data_transforms.random_crop_to_size(
                mode_cfg.crop_size,
                crop_feats,
                seed=ensemble_seed + 1,
            )
        )
        transforms.append(
            data_transforms.make_fixed_size(
                crop_feats,
                mode_cfg.crop_size,
            )
        )
    
    return transforms


def process_tensors_from_config(
    tensors: TensorDict, 
    common_cfg, 
    mode_cfg
) -> TensorDict:
    """Based on the config, apply filters and transformations to the data."""
    ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg, 
            mode_cfg, 
            ensemble_seed,
        )
        fn = compose(fns)
        # d["ensemble_index"] = i
        return fn(d)
    
    nonensembled = nonensembled_transform_fns(mode_cfg)

    tensors = compose(nonensembled)(tensors)

    if "no_recycling_iters" in tensors:
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )

    return tensors


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict
