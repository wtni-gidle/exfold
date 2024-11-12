from typing import Dict
import argparse
import time
from functools import partial
import json
import logging
import os
from multiprocessing import Pool, Manager

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from exfold.data.mmcif_parsing import parse


def parse_file(cif_path, results: Dict):
    file_id = os.path.splitext(os.path.basename(cif_path))[0]
    with open(cif_path, "r") as fp:
        mmcif_string = fp.read()
    result = parse(file_id=file_id, mmcif_string=mmcif_string)

    if result.mmcif_object is None:
        error = list(result.errors.values())[0]
        logging.error(f'Failed to parse {file_id}. Skipping...')
        logging.error(f'{error.__class__.__name__}: {error}')
        return

    result = result.mmcif_object

    local_data = {}
    local_data["header"] = result.header
    local_data["protein"] = result.chain_to_seqres["protein"]
    local_data["rna"] = result.chain_to_seqres["rna"]
    local_data["dna"] = result.chain_to_seqres["dna"]
    local_data["mmcif_to_custom_mapping"] = result.mmcif_to_custom_mapping
    local_data["no_chains"] = {k: len(v) for k, v in result.chain_to_seqres.items()}
    
    results.update({file_id: local_data})


def main(args):
    logging.basicConfig(level=logging.WARNING, handlers=[logging.FileHandler(args.log_path)])
    
    start = time.perf_counter()
    cif_files = [os.path.join(args.mmcif_dir, file) for file in os.listdir(args.mmcif_dir)]

    results = Manager().dict()
    fn = partial(parse_file, results=results)

    with Pool(processes=args.no_workers) as pool:
        pool.map(fn, cif_files, chunksize=args.chunksize)
    
    with open(args.cache_path, "w") as fp:
        json.dump(dict(results), fp, indent=4)
    logging.warning(f"CIF cache saved in {args.cache_path}")

    total_time = time.perf_counter() - start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.warning(f"Total time: {int(hours)} h {int(minutes)} min {seconds:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mmcif_dir", type=str,
        help="Path to directory containing mmCIF files"
    )
    parser.add_argument(
        "cache_path", type=str,
        help="Path for .json output"
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
