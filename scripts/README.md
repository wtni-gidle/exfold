# Step

1. download_pdb_mmcif

    ```shell
    sh download_pdb_mmcif.sh /work/hdd/bbgs/nwentao/data
    ```

2. generate_mmcif_cache

    ```shell
    sbatch generate_mmcif_cache.sh
    ```

3. generate_train_val_fasta

    ```shell
    python generate_train_val_fasta.py \
        /work/hdd/bbgs/nwentao/data/exfold/mmcif_cache.json \
        /work/hdd/bbgs/nwentao/data/exfold/generate_mmcif_cache.log \
        /work/hdd/bbgs/nwentao/data/exfold/train_val.fasta \
        /work/hdd/bbgs/nwentao/data/exfold/generate_train_val_fasta.log
    ```

4. cluster_and_split

    ```shell
    python cluster_and_split.py \
        /work/hdd/bbgs/nwentao/data/exfold/train_val.fasta \
        /work/hdd/bbgs/nwentao/data/exfold/clusters.txt \
        /work/hdd/bbgs/nwentao/data/exfold/train_raw.txt \
        /work/hdd/bbgs/nwentao/data/exfold/val_raw.txt \
        --n_cpu 32
    ```
