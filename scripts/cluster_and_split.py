import subprocess
import os
import tempfile
import re
import random
import argparse


def run_cd_hit(input_fasta, output_file, n_cpu):
    cmd = [
        "cd-hit-est", 
        "-i", input_fasta,
        "-o", output_file,
        "-c", "0.8",
        "-n", "5",
        "-M", "0",
        "-T", str(n_cpu),
    ]
    subprocess.run(cmd, shell=True, check=True)


def parse_clusters(text):
    clusters = {}
    current_cluster = None

    for line in text.splitlines():
        cluster_match = re.match(r"^>Cluster (\d+)", line)
        if cluster_match:
            current_cluster = f"Cluster {cluster_match.group(1)}"
            clusters[current_cluster] = []
        elif current_cluster:
            sequence_match = re.search(r">(\S+)", line)
            if sequence_match:
                sequence_name = sequence_match.group(1)
                clusters[current_cluster].append(sequence_name[:-3])

    return clusters


def main(args):
    random.seed(42)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "cluster.fasta")
        run_cd_hit(args.input_fasta, output_file, args.n_cpu)
        with open(f"{output_file}.clstr", "r") as f:
            clusters = parse_clusters(f.read())
    clusters = list(clusters.values())
    with open(args.cluster_file, "w") as f:
        for cluster in clusters:
            f.write(",".join(cluster) + "\n")
    val_clusters = random.sample(clusters, int(len(clusters) * 0.1))
    train_clusters = [c for c in clusters if c not in val_clusters]

    with open(args.train_path, "w") as f:
        for cluster in train_clusters:
            f.write("\n".join(cluster) + "\n")
    with open(args.val_path, "w") as f:
        for cluster in val_clusters:
            f.write("\n".join(cluster) + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fasta", type=str)
    parser.add_argument("cluster_file", type=str)
    parser.add_argument("train_path", type=str)
    parser.add_argument("val_path", type=str)
    parser.add_argument("--n_cpu", type=int, default=4)
    args = parser.parse_args()
    main(args)
