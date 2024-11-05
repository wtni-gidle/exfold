from typing import Dict
import copy
from functools import partial
import json
import logging
import os
import pickle
from typing import Optional, Sequence, Any, Union
import numpy as np

import ml_collections as mlc
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from exfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
)
from exfold.utils.tensor_utils import dict_multimap
from exfold.utils.tensor_utils import (
    tensor_tree_map,
)

FeatureDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


class ExFoldSingleDataset(Dataset):
    def __init__(
        self,
        config: mlc.ConfigDict,
        data_dir: str,
        ss_dir: str,
        filter_path: str,
        chain_data_cache_path: Optional[str] = None,
        lm_emb_dir: Optional[str] = None,
        mode: str = "train",
        _output_raw: bool = False,
    ):
        """
        Args:
            data_dir:
                A path to a directory containing mmCIF files (in train
                mode) or FASTA files (in inference mode).
            filter_path:
                record the chain ids to be used
            config:
                A dataset config object. See exfold.config
        """
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.ss_dir = ss_dir
        self.lm_emb_dir = lm_emb_dir
        self.mode = mode
        self._output_raw = _output_raw

        # todo mode的具体值还需要统一检查一下
        valid_modes = ["train", "eval"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')
        
        with open(filter_path, "r") as f:
            self._chain_ids = [line.strip() for line in f.readlines()]
        
        self.chain_data_cache = None
        if chain_data_cache_path is not None:
            with open(chain_data_cache_path, "r") as fp:
                self.chain_data_cache = json.load(fp)
                # chain ids in filter_path must be recorded in chain_data_cache
                assert set(self._chain_ids).issubset(set(self.chain_data_cache.keys()))
        
        self._chain_id_to_idx_dict = {
            data: i for i, data in enumerate(self._chain_ids)
        }
        
        self.data_pipeline = data_pipeline.DataPipeline()

        if not self._output_raw:
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config)
    
    def _parse_mmcif(
        self, 
        path: str, 
        file_id: str, 
        chain_id: str, 
        ss_dir: str,
        ss_methods: Sequence[str]
    ) -> FeatureDict:
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            chain_id=chain_id,
            ss_dir=ss_dir,
            ss_methods=ss_methods,
        )

        return data

    def chain_id_to_idx(self, chain_id: str):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx: int):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        """
        Get the file path, pass it sequentially into `data_pipeline` 
        and `feature_pipeline` to generate `FeatureDict`.
        """
        # {file_id}_{chain_id}
        name = self.idx_to_chain_id(idx)
        ss_dir = os.path.join(self.ss_dir, name)

        file_id, chain_id = name.rsplit('_', 1)
        
        path = os.path.join(self.data_dir, f"{file_id}.cif")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")
        
        data = self._parse_mmcif(
            path=path, 
            file_id=file_id, 
            chain_id=chain_id, 
            ss_dir=ss_dir, 
            ss_methods=self.config.ss_methods
        )

        if self._output_raw:
            return data
        
        feats = self.feature_pipeline.process_features(
            data, self.mode
        )

        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["restype"].shape[-1])],
            dtype=torch.int64,
            device=feats["restype"].device)

        return feats
    
    def __len__(self):
        return len(self._chain_ids)


class ExFoldDataset(Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """
    def __init__(
        self,
        datasets: Sequence[ExFoldSingleDataset],
        probabilities: Sequence[float],
        epoch_len: int,
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator

        self._samples = [self.looped_samples(i) for i in range(len(self.datasets))]
        if _roll_at_init:
            self.reroll()
    
    @staticmethod
    def get_stochastic_train_filter_prob(cache_entry: Dict) -> float:
        # Stochastic filters
        # 目前设置为1.0，即不使用随机过滤
        # 以后可以考虑af2的随机过滤，以及根据链是否和其他链有大量相互作用来调整概率
        return 1.0
    
    def looped_shuffled_dataset_idx(self, dataset_len: int):
        """Generate an infinite loop of shuffled indices in a dataset."""
        while True:
            # Uniformly shuffle each dataset's indices
            weights = [1. for _ in range(dataset_len)]
            shuf = torch.multinomial(
                torch.tensor(weights),
                num_samples=dataset_len,
                replacement=False,
                generator=self.generator,
            )
            for idx in shuf:
                yield idx

    def looped_samples(self, dataset_idx: int):
        """Generate an infinite loop of samples from the specified dataset with stochastic filtering."""
        max_cache_len = int(self.epoch_len * self.probabilities[dataset_idx])
        dataset = self.datasets[dataset_idx]
        idx_iter = self.looped_shuffled_dataset_idx(len(dataset))
        chain_data_cache = dataset.chain_data_cache
        while True:
            weights = []
            idx = []
            for _ in range(max_cache_len):
                candidate_idx = next(idx_iter)
                chain_id = dataset.idx_to_chain_id(candidate_idx)
                chain_data_cache_entry = chain_data_cache[chain_id]

                p = self.get_stochastic_train_filter_prob(
                    chain_data_cache_entry,
                )
                weights.append([1. - p, p])
                idx.append(candidate_idx)

            samples = torch.multinomial(
                torch.tensor(weights),
                num_samples=1,
                generator=self.generator,
            )
            samples = samples.squeeze()

            cache = [i for i, s in zip(idx, samples) if s]

            for datapoint_idx in cache:
                yield datapoint_idx
    
    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]
    
    def __len__(self):
        return self.epoch_len

    def reroll(self):
        """Generate `datapoints` for the next epoch."""
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )
        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class ExFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)


class ExFoldDataLoader(DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()
    
    def _prep_batch_properties_probs(self):
        """Generate `recycling_probs_tensor`."""
        stage_cfg = self.config[self.stage]
        max_iters = self.config.common.max_recycling_iters

        if stage_cfg.uniform_recycling:
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        self.recycling_probs_tensor = torch.tensor(
            recycling_probs, 
            dtype=torch.float32
        )
    
    def _sample_no_recycling_iters(self, batch: TensorDict) -> TensorDict: 
        """Sample the number of recycling iterations and update the batch accordingly."""
        no_recycling = int(torch.multinomial(
            self.recycling_probs_tensor,
            num_samples=1,
            generator=self.generator
        )[0])

        # add `no_recycling_iters`
        restype = batch["restype"]
        batch_dims = restype.shape[:-2]
        recycling_dim = restype.shape[-1]
        batch["no_recycling_iters"] = torch.full(
            size=batch_dims + (recycling_dim,), 
            fill_value=no_recycling,
            device=restype.device,
            requires_grad=False,
        )

        # select dims according to `no_recycling_iters`
        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)

        return batch
    
    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._sample_no_recycling_iters(batch)

        return _batch_prop_gen(it)


#* only support `train`, `validate` and `test`
class ExFoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: mlc.ConfigDict,
        train_data_dir: Optional[str] = None,
        train_ss_dir: Optional[str] = None,
        train_filter_path: Optional[str] = None,
        train_chain_data_cache_path: Optional[str] = None,
        distillation_data_dir: Optional[str] = None,
        distillation_ss_dir: Optional[str] = None,
        distillation_filter_path: Optional[str] = None,
        distillation_chain_data_cache_path: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        val_ss_dir: Optional[str] = None,
        val_filter_path: Optional[str] = None,
        test_data_dir: Optional[str] = None,
        test_ss_dir: Optional[str] = None,
        test_filter_path: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: int = 50000,
        **kwargs
    ):
        super().__init__()
        #todo: need to check dataset args of different modes
        self.config = config

        self.train_data_dir = train_data_dir
        self.train_ss_dir = train_ss_dir
        self.train_filter_path = train_filter_path
        self.train_chain_data_cache_path = train_chain_data_cache_path

        self.distillation_data_dir = distillation_data_dir
        self.distillation_ss_dir = distillation_ss_dir
        self.distillation_filter_path = distillation_filter_path
        self.distillation_chain_data_cache_path = distillation_chain_data_cache_path

        self.val_data_dir = val_data_dir
        self.val_ss_dir = val_ss_dir
        self.val_filter_path = val_filter_path

        self.test_data_dir = test_data_dir
        self.test_ss_dir = test_ss_dir
        self.test_filter_path = test_filter_path

        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        self.training_mode = self.train_data_dir is not None

        if self.training_mode and test_data_dir is not None:
            raise ValueError("In training mode, test_data_dir must be None.")
    
    def setup(self, stage=None):
        dataset_gen = partial(
            ExFoldSingleDataset,
            config=self.config,
        )

        if self.training_mode:
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                ss_dir=self.train_ss_dir,
                filter_path=self.train_filter_path,
                chain_data_cache_path=self.train_chain_data_cache_path,
                mode="train",
            )

            distillation_dataset = None
            if self.distillation_data_dir is not None:
                distillation_dataset = dataset_gen(
                    data_dir=self.distillation_data_dir,
                    ss_dir=self.distillation_ss_dir,
                    filter_path=self.distillation_filter_path,
                    chain_data_cache_path=self.distillation_chain_data_cache_path,
                    mode="train",
                )
            
            if distillation_dataset is not None:
                datasets = [train_dataset, distillation_dataset]
                d_prob = self.config.train.distillation_prob
                probabilities = [1. - d_prob, d_prob]
            else:
                datasets = [train_dataset]
                probabilities = [1.]
            
            generator = None
            if self.batch_seed is not None:
                generator = torch.Generator()
                generator = generator.manual_seed(self.batch_seed + 1)
            
            self.train_dataset = ExFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=self.train_epoch_len,
                generator=generator,
                _roll_at_init=False,
            )
            
            if self.val_data_dir is not None:
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    ss_dir=self.val_ss_dir,
                    filter_path=self.val_filter_path,
                    mode="eval",
                )
            
            else:
                self.eval_dataset = None
        else:
            self.test_dataset = dataset_gen(
                data_dir=self.test_data_dir,
                ss_dir=self.test_ss_dir,
                filter_path=self.test_filter_path,
                mode="eval", # todo: 看一下这个有没有问题
            )
    
    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)
        
        if stage == "train":
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid stage: {stage}")

        batch_collator = ExFoldBatchCollator()

        dl = ExFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl
    
    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return None
    
    def test_dataloader(self):
        return self._gen_dataloader("test")
