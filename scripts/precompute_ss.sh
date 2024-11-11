#!/bin/bash
#SBATCH --job-name="precompute_ss"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output="precompute_ss.%j.%N.out"
#SBATCH --error="precompute_ss.%j.%N.out"
#SBATCH --account=bbgs-delta-cpu
#SBATCH --export=ALL
#SBATCH -t 04:00:00

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

fasta_path="/work/hdd/bbgs/nwentao/data/exfold/train_val.fasta"
output_dir="/work/hdd/bbgs/nwentao/data/exfold/ss"
log_path="/work/hdd/bbgs/nwentao/data/exfold/precompute_ss.log"

python scripts/precompute_ss.py \
    $fasta_path \
    $output_dir \
    $log_path \
    --no_workers 64 \

echo "Done"