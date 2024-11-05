import json
import logging
import argparse

logging.basicConfig(level=logging.INFO)

DATE_CUTOFF = "2021-01-01"

def deduplicate_dict(input_dict):
    seen_values = set()
    result_dict = {}

    for key, value in input_dict.items():
        if value not in seen_values:
            result_dict[key] = value
            seen_values.add(value)

    return result_dict


def write_fasta(file_path, sequences):
    with open(file_path, 'w') as fasta_file:
        for name, seq in sequences.items():
            fasta_file.write(f">{name}\n")
            fasta_file.write(f"{seq}\n")


def main(args):
    with open(args.log_path, "r") as f:
        cache_log = f.read().splitlines()

    error_num = 0
    for line in cache_log:
        if "Failed to parse" in line:
            error_num += 1
    logging.info(f"Number of files that failed to parse: {error_num}")

    hybrid_num = 0
    for line in cache_log:
        if "Hybrid" in line:
            hybrid_num += 1
    logging.info(f"Number of hybrid errors: {hybrid_num}")

    other_error_num = error_num - hybrid_num
    logging.info(f"Number of other errors: {other_error_num}")

    with open(args.cache_path, "r") as f:
        cache = json.load(f)
    logging.info(f"Number of total mmcif files: {error_num + len(cache)}")
    logging.info(f"Number of parsed mmcif files: {len(cache)}")

    # has rna
    new_cache = {}
    for file_id, info in cache.items():
        if info["no_chains"]["rna"] > 0:
            new_cache[file_id] = info
    logging.info(f"Number of mmcif files with RNA: {len(new_cache)}")

    # resolution <= 4.5
    tmp = {}
    for file_id, info in new_cache.items():
        if info["header"]["resolution"] <= 4.5:
            tmp[file_id] = info
    new_cache = tmp
    logging.info(f"Number of mmcif files with resolution <= 4.5: {len(new_cache)}")

    # release date < 2021-01-01
    tmp = {}
    for file_id, info in new_cache.items():
        release_date = info["header"]["release_date"]
        if (release_date == "?") or (release_date < DATE_CUTOFF):
            tmp[file_id] = info
    new_cache = tmp
    logging.info(f"Number of mmcif files with release date before {DATE_CUTOFF}: {len(new_cache)}")

    # extract rna chains
    rna_chains = {}
    for file_id, info in new_cache.items():
        # keep the first chain only for identical chains in each file
        for k, v in deduplicate_dict(info["rna"]).items():
            rna_chains[f"{file_id}_{k}"] = v
    print(f"Number of RNA chains: {len(rna_chains)}")

    # length >= 15
    tmp = {}
    for chain_id, seq in rna_chains.items():
        if len(seq) >= 15:
            tmp[chain_id] = seq
    print(f"Number of RNA chains with length >= 15: {len(tmp)}, remove {len(rna_chains) - len(tmp)}")
    rna_chains = tmp

    # unk to X
    for chain_id, seq in rna_chains.items():
        seq = ''.join(['X' if nucleotide not in 'AUCG' else nucleotide for nucleotide in seq])
        rna_chains[chain_id] = seq

    # < 90% identical
    tmp = {}
    for chain_id, seq in rna_chains.items():
        s = max(seq, key=seq.count)
        prop = seq.count(s) / len(seq)
        if prop < 0.9:
            tmp[chain_id] = seq
    print(f"Number of RNA chains with <90% identical nucleotides: {len(tmp)}, remove {len(rna_chains) - len(tmp)}")
    rna_chains = tmp

    # < 5% X
    tmp = {}
    for chain_id, seq in rna_chains.items():
        prop = seq.count("X") / len(seq)
        if prop < 0.05:
            tmp[chain_id] = seq
    print(f"Number of RNA chains with <5% X: {len(tmp)}, remove {len(rna_chains) - len(tmp)}")
    rna_chains = tmp

    logging.info(f"Writing fasta to {args.fasta_path}")
    write_fasta(args.fasta_path, rna_chains)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cache_path", type=str,
        help="Path for .json mmcif cache"
    )
    parser.add_argument(
        "log_path", type=str,
        help="Path for .json mmcif cache log"
    )
    parser.add_argument(
        "fasta_path", type=str,
        help="Path for .fasta output"
    )
    args = parser.parse_args()

    main(args)
