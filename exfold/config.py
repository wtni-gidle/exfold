import copy
import importlib
import ml_collections as mlc


c_z = mlc.FieldReference(64, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(64, field_type=int)

num_ss = mlc.FieldReference(2, field_type=int)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
e2e_distogram_bins = mlc.FieldReference(36+2, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)
loss_unit_distance = mlc.FieldReference(10.0, field_type=float) # trans_scale_factor
use_high_precision = mlc.FieldReference(True, field_type=bool)
masked_msa_dim = mlc.FieldReference(4+2, field_type=int)


NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"


basic_config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "restype": [NUM_RES],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
                    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
                    "true_msa": [NUM_MSA_SEQ, NUM_RES],
                    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
                    "ss": [NUM_RES, NUM_RES, None],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "no_recycling_iters": [],
                    "use_clamped_fape": [],
                    "is_distillation": [],
                },
                "masked_msa": {
                    "same_prob": 0.15,
                    "uniform_prob": 0.15,
                },
                "max_recycling_iters": 3,
                "unsupervised_features": [
                    "restype",
                    "residue_index",
                    "msa",
                    "ss",
                    "seq_length",
                    "no_recycling_iters",
                ],
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_positions",
                    "use_clamped_fape",
                    "is_distillation",
                ],
            },
            "predict": {
                "fixed_size": True,
                # AF2: Note that this masking is used both at training time, and at inference time.
                "masked_msa_replace_fraction": 0.15, # todo 可以调整看看在inference的时候不用mask会不会影响效果
                "crop": False,
                "crop_size": None,
                "supervised": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                # AF2: Note that this masking is used both at training time, and at inference time.
                "masked_msa_replace_fraction": 0.15, # todo 可以调整看看在inference的时候不用mask会不会影响效果
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "masked_msa_replace_fraction": 0.15,
                "crop": True,
                "crop_size": 256,
                "supervised": True,
                "clamp_prob": 0.9,
                "uniform_recycling": True,
                "distillation_prob": 0.75,
            },
            "data_module": {
                "data_loaders": {
                    "batch_size": 1,    # todo
                    "num_workers": 16,
                    "pin_memory": True,
                },
            },
        },
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            # Use DeepSpeed memory-efficient attention kernel. Mutually
            # exclusive with use_lma and use_flash.
            "use_deepspeed_evo_attention": False,
            # Use Staats & Rabe's low-memory attention algorithm. Mutually
            # exclusive with use_deepspeed_evo_attention and use_flash.
            "use_lma": False,
            # Use FlashAttention in selected modules. Mutually exclusive with 
            # use_deepspeed_evo_attention and use_lma. Doesn't work that well
            # on long sequences (>1000 residues).
            "use_flash": False,
            "offload_inference": False,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
            "is_combined": False, #todo 需要设置检查条件, 有且仅有一个为True
            "is_e2e": False,
            "is_geom": False,
        },
        "model": {
            "_mask_trans": False,
            "seq_embedder": {
                "tf_dim": 5,
                "msa_dim": 6,
                "c_z": c_z,
                "c_m": c_m,
                "relpos_k": 64,
            },
            "ss_embedder": {
                "ss_dim": 2*num_ss, 
                "c_z": c_z,
            },
            "recycling_embedder": {
                "c_m": c_m,
                "c_z": c_z,
                "min_bin": 2.0,
                "max_bin": 40.0,
                "no_bins": 20,
                "inf": 1e8,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": 16,
                "c_hidden_opm": 16,
                "c_hidden_mul": c_z,
                "c_hidden_pair_att": 16,
                "c_s": c_s,
                "no_heads_msa": 8,
                "no_heads_pair": 8,
                "no_blocks": 48,
                "transition_n": 2,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "no_column_attention": True,
                "opm_first": False,
                "fuse_projection_weights": False,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
                "inf": 1e9,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "no_heads_ipa": 8,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 4,
                "no_transition_layers": 1,
                "trans_scale_factor": loss_unit_distance,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
                "use_high_precision": use_high_precision
            },
            "e2e_heads": {
                "distogram": {
                    "c_z": c_z,
                    "no_bins": e2e_distogram_bins,
                },
                "masked_msa": {
                    "c_m": c_m,
                    "c_out": masked_msa_dim,
                },
            },
            "geom_heads": {
                "PP": { # P(i)-P(j) distance
                    "c_z": c_z,
                    "c_hidden": c_z,
                    "no_blocks": 1,
                    "no_bins": 56+2,
                    "symmetrize": True,
                },
                "CC": { # C4'(i)-C4'(j) distance
                    "c_z": c_z,
                    "c_hidden": c_z,
                    "no_blocks": 1,
                    "no_bins": 44+2,
                    "symmetrize": True,
                },
                "NN": { # N(i)-N(j) distance
                    "c_z": c_z,
                    "c_hidden": c_z,
                    "no_blocks": 1,
                    "no_bins": 32+2,
                    "symmetrize": True,
                },
                "PCCP": { # P(i)-C4'(i)-C4'(j)-P(j) distance
                    "c_z": c_z,
                    "c_hidden": c_z,
                    "no_blocks": 1,
                    "no_bins": 36+1,
                    "symmetrize": True,
                },
                "CNNC": { # C4'(i)-N(i)-N(j)-C4'(j) distance
                    "c_z": c_z,
                    "c_hidden": c_z,
                    "no_blocks": 1,
                    "no_bins": 36+1,
                    "symmetrize": True,
                },
                "PNNP": { # P(i)-N(i)-N(j)-P(j) distance
                    "c_z": c_z,
                    "c_hidden": c_z,
                    "no_blocks": 1,
                    "no_bins": 36+1,
                    "symmetrize": True,
                },
                "masked_msa": {
                    "c_m": c_m,
                    "c_out": masked_msa_dim,
                },
            }
        },
        "e2e_loss": {
            "distogram": {
                "min_bin": 2.0,
                "max_bin": 40.0,
                "no_bins": e2e_distogram_bins,
                "eps": eps, 
                "weight": 0.6,
            },
            "fape": {
                "intermediate": {
                    "clamp_distance": 30.0,
                    "loss_unit_distance": loss_unit_distance,
                    "weight": 0.5,
                },
                "final": {
                    "clamp_distance": 30.0,
                    "loss_unit_distance": loss_unit_distance,
                    "weight": 0.5,
                },
                "weight": 1.5,
            },
            "masked_msa": {
                "num_classes": masked_msa_dim, 
                "eps": eps,  # 1e-8,
                "weight": 1.0,
            },
        },
        "geom_loss": {
            "PP": {
                "min_bin": 2.0,
                "max_bin": 30.0,
                "no_bins": 56+2,
                "eps": eps,
                "weight": 1.0,
            },
            "CC": {
                "min_bin": 2.0,
                "max_bin": 24.0,
                "no_bins": 44+2,
                "eps": eps,
                "weight": 1.0,
            },
            "NN": {
                "min_bin": 2.0,
                "max_bin": 18.0,
                "no_bins": 32+2,
                "eps": eps,
                "weight": 1.0,
            },
            "PCCP": {
                "min_bin": -180,
                "max_bin": 180,
                "no_bins": 36+1,
                "max_dist": 24,
                "eps": eps,
                "weight": 0.5,
            },
            "PNNP": {
                "min_bin": -180,
                "max_bin": 180,
                "no_bins": 36+1,
                "max_dist": 18,
                "eps": eps,
                "weight": 0.5,
            },
            "CNNC": {
                "min_bin": -180,
                "max_bin": 180,
                "no_bins": 36+1,
                "max_dist": 18,
                "eps": eps,
                "weight": 0.5,
            },
            "masked_msa": {
                "num_classes": masked_msa_dim, 
                "eps": eps,
                "weight": 1.0,
            },
        },
        "ema": {"decay": 0.999},
        "train": {
            "optimizer": {
                "lr": 0.001,   # need to check lr equal to base_lr in lr_scheduler
                "betas": (0.9, 0.999),
                "eps": 1e-6
            },
            "lr_scheduler": {
                "base_lr": 0.0,
                "max_lr": 0.001,
                "warmup_steps": 1000,
                "start_decay_after_n_steps": 50000,
                "decay_every_n_steps": 50000,
                "decay_factor": 0.95,
            }
        },
        
    }
)

