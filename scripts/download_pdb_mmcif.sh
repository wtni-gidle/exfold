#!/bin/bash
############################################################################
# Script for PDB download
# -----------------------
# 下载所有mmcif文件
# 使用方法：download_pdb_mmcif.sh /path/to/data_dir
############################################################################

set -e

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/pdb_mmcif"
RAW_DIR="${ROOT_DIR}/raw"
MMCIF_DIR="${ROOT_DIR}/mmcif_files"
LOG_FILE="${ROOT_DIR}/log.stdout"

mkdir -p "${RAW_DIR}"

echo "Running rsync to fetch all mmCIF files (note that the rsync progress estimate might be inaccurate)..."
rsync -rlpt --info=progress2 --compress --delete --port=33444 rsync.wwpdb.org::ftp/data/structures/divided/mmCIF/ \
"${RAW_DIR}" 1> "${LOG_FILE}" 2>&1

echo "Unzipping all mmCIF files..."
find "${RAW_DIR}/" -type f -iname "*.gz" -exec gunzip {} +

echo "Flattening all mmCIF files..."
mkdir -p "${MMCIF_DIR}"
find "${RAW_DIR}" -type d -empty -delete  # Delete empty directories.
for subdir in "${RAW_DIR}"/*
do
    mv "${subdir}/"*.cif "${MMCIF_DIR}"
done

# Delete empty download directory structure.
find "${RAW_DIR}" -type d -empty -delete

echo "Done"