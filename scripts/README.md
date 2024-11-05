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
    /work/hhd/bbgs/nwentao/data/exfold/mmcif_cache.json \
    /work/hhd/bbgs/nwentao/data/exfold/mmcif_cache_log.json \
    /work/hhd/bbgs/nwentao/data/exfold/train_val.fasta
```
