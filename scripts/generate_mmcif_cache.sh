#!/bin/bash
#SBATCH --job-name="generate_mmcif_cache"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output="generate_mmcif_cache.%j.%N.out"
#SBATCH --error="generate_mmcif_cache.%j.%N.out"
#SBATCH --account=bbgs-delta-cpu
#SBATCH --export=ALL
#SBATCH -t 02:00:00

# make the script stop when error (non-true exit code) occurs
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

conda activate pldev
cd /work/hdd/bbgs/nwentao/projects/exfold

mkdir -p /work/hdd/bbgs/nwentao/data/exfold

mmcif_dir="/work/hdd/bbgs/nwentao/data/pdb_mmcif/mmcif_files/"
output_path="/work/hdd/bbgs/nwentao/data/exfold/mmcif_cache.json"
log_path="/work/hdd/bbgs/nwentao/data/exfold/mmcif_cache_log.json"

python scripts/generate_mmcif_cache.py \
    "$mmcif_dir" \
    "$output_path" \
    "$log_path" \
    --no_workers 64 \

echo "Done"