# Step

1. download_pdb_mmcif

```shell
sh download_pdb_mmcif.sh /work/hdd/bbgs/nwentao/data
```

1. generate_mmcif_cache

```shell
sbatch generate_mmcif_cache.sh
```

1. generate_train_val_fasta

```shell
python generate_train_val_fasta.py \
    /work/hdd/bbgs/nwentao/data/exfold/mmcif_cache.json \
    /work/hdd/bbgs/nwentao/data/exfold/generate_mmcif_cache.log \
    /work/hdd/bbgs/nwentao/data/exfold/train_val.fasta \
    /work/hdd/bbgs/nwentao/data/exfold/generate_train_val_fasta.log
```
