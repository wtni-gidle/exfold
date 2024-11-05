import copy
from typing import Tuple, List, Dict, Sequence

import ml_collections
import numpy as np
import torch

from exfold.data import input_pipeline


FeatureDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def np_to_tensor_dict(
    np_example: Dict[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    # torch generates warnings if feature is already a torch Tensor
    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
    tensor_dict = {
        k: to_tensor(v) for k, v in np_example.items() if k in features
    }

    return tensor_dict


# todo lm_emb mode
def make_data_config(
    config: ml_collections.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res

    feature_names = cfg.common.unsupervised_features

    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def np_example_to_features(
    np_example: FeatureDict,
    config: ml_collections.ConfigDict,
    mode: str
) -> TensorDict:
    # Retain only the necessary features from the FeatureDict (data_pipeline output)
    # and convert them into tensors
    seq_length = np_example["seq_length"]
    num_res = int(seq_length[0]) if seq_length.ndim != 0 else int(seq_length)
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)
    
    tensor_dict = np_to_tensor_dict(
        np_example=np_example, features=feature_names
    )

    # I moved this feature in front of the input pipeline
    # use_clamped_fape is not set in predict mode
    if "use_clamped_fape" in feature_names:
        if mode == "train":
            # 1.11.5 Loss clamping details
            p = torch.rand(1).item()
            use_clamped_fape_value = float(p < cfg.supervised.clamp_prob)
            tensor_dict["use_clamped_fape"] = torch.full(
                size=[num_res],
                fill_value=use_clamped_fape_value,
                dtype=torch.float32,
            )
        else:
            features["use_clamped_fape"] = torch.full(
                size=[num_res],
                fill_value=0.0,
                dtype=torch.float32,
            )

    with torch.no_grad():
        features = input_pipeline.process_tensors_from_config(
            tensor_dict,
            cfg.common,
            cfg[mode],
        )

    return {k: v for k, v in features.items()}
    

class FeaturePipeline:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ):
        self.config = config

    def process_features(
        self,
        raw_features: FeatureDict,
        mode: str = "train",
    ) -> TensorDict:
        return np_example_to_features(
            np_example=raw_features,
            config=self.config,
            mode=mode,
        )
