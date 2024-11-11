import argparse
import time
from functools import partial
import logging
import os
from multiprocessing import Pool
import tempfile
from Bio import SeqIO
from typing import Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from exfold.data.data_pipeline import SSRunner


RNAFOLD_BINARY_PATH = "/work/hdd/bbgs/nwentao/tools/ss_tools/ViennaRNA/2.4.18/bin/RNAfold"
PETFOLD_BINARY_PATH = "/work/hdd/bbgs/nwentao/tools/PETfold/bin/PETfold"


def run_ss(desc_seq_pair: Tuple[str, str], ss_runner: SSRunner, output_dir):
    desc, seq = desc_seq_pair
    ss_dir = os.path.join(output_dir, desc)
    os.makedirs(ss_dir, exist_ok=True)
    
    fd, fasta_path = tempfile.mkstemp(suffix=".fasta")
    with os.fdopen(fd, 'w') as fp:
        fp.write(f'>seq\n{seq}')
    
    try:
        ss_runner.run(
            fasta_path=fasta_path,
            output_dir=ss_dir,
        )
    except Exception as e:
        logging.error(e)
        logging.error(f"Failed to run ss for {desc}. Skipping...")
        os.remove(fasta_path)
        os.rmdir(ss_dir)
        return

    os.remove(fasta_path)


def main(args):
    logging.basicConfig(level=logging.WARNING, handlers=[logging.FileHandler(args.log_path)])
    
    logging.warning(f"Start precomputing ss, logs saved in {args.log_path}...")
    start = time.perf_counter()

    desc_seq_map = {record.id: str(record.seq) for record in SeqIO.parse(args.fasta_path, "fasta")}
    desc_seq_pairs = list(desc_seq_map.items())
    ss_runner = SSRunner(
        rnafold_binary_path=RNAFOLD_BINARY_PATH,
        petfold_binary_path=PETFOLD_BINARY_PATH,
    )
    fn = partial(run_ss, ss_runner=ss_runner, output_dir=args.output_dir)

    with Pool(args.num_workers) as pool:
        pool.map(fn, desc_seq_pairs, chunksize=args.chunksize)
    
    total_time = time.perf_counter() - start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.warning(f"Total time: {int(hours)} h {int(minutes)} min {seconds:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path", type=str,
        help="Path for .fasta input"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Directory in which to output ss"
    )
    parser.add_argument(
        "log_path", type=str,
        help="Path for .log log"
    )
    parser.add_argument(
        "--no_workers", type=int, default=4,
        help="Number of workers to use for parsing"
    )
    parser.add_argument(
        "--chunksize", type=int, default=10,
        help="How many files should be distributed to each worker at a time"
    )

    args = parser.parse_args()

    main(args)
