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
        fp.write(f'>seq\n{seq}\n')
    
    ss_runner.run(
        fasta_path=fasta_path,
        output_dir=ss_dir,
    )
    # try:
    #     ss_runner.run(
    #         fasta_path=fasta_path,
    #         output_dir=ss_dir,
    #     )
    # except Exception as e:
    #     logging.error(e)
    #     logging.error(f"Failed to run ss for {desc}. Skipping...")
    #     os.remove(fasta_path)
    #     os.rmdir(ss_dir)
    #     return

    os.remove(fasta_path)


def main():
    logging.basicConfig(level=logging.INFO)
    fasta_path="test.fasta"
    output_dir="ss"

    desc_seq_map = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
    desc_seq_pair = list(desc_seq_map.items())[0]
    ss_runner = SSRunner(
        rnafold_binary_path=RNAFOLD_BINARY_PATH,
        petfold_binary_path=PETFOLD_BINARY_PATH,
    )
    fn = partial(run_ss, ss_runner=ss_runner, output_dir=output_dir)
    fn(desc_seq_pair)


if __name__ == "__main__":
    main()
