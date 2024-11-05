raw data (from data_pipeline):

> fasta:
sequence_features:
    restype: [N_res, 4+1]
    residue_index: [N_res]
    seq_length: [N_res]
msa_features:
    msa: [1, N_res]
ss_features:
    ss: [N_res, N_res, 2*N_method]

> mmcif:
mmcif_features:
    restype: [N_res, 4+1]
    residue_index: [N_res]
    seq_length: [N_res]
    all_atom_positions: [N_res, rna_backbone_atom_num, 3]
    all_atom_mask: [N_res, rna_backbone_atom_num]
    is_distillation: []
msa_features:
    msa: [1, N_res] no gap since no msa
ss_features:
    ss: [N_res, N_res, 2*N_method]


processed data (from feature_pipeline):
the final dimension is `max_recycling_iters`+1 (而后会被裁剪)
> mmcif:
restype: [N_res]
residue_index: [N_res]
seq_length: []
seq_mask: [N_res]
target_feat: [N_res, 4+1]

backbone_rigid_tensor: [N_res, 4, 4]
backbone_rigid_mask: [N_res]

msa_mask: [N_seq, N_res]
bert_mask: [N_seq, N_res] 1: masked
true_msa: [N_seq, N_res] gt msa
msa_feat: [N_seq, N_res, 4+2]

ss: [N_res, N_res, 2*N_method]

all_atom_positions: [N_res, rna_backbone_atom_num, 3]
all_atom_mask: [N_res, rna_backbone_atom_num]
use_clamped_fape: []
is_distillation: []

no_recycling_iters: [] 实际的recycle次数，可以取到0

