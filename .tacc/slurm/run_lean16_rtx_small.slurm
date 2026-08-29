#!/bin/bash
#SBATCH -J vipe-lean16-rtx
#SBATCH -A IRI26004
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH -t 24:00:00
#SBATCH -o .tacc/logs/%x-%j.out
#SBATCH -e .tacc/logs/%x-%j.err

set -eo pipefail
module load tacc-apptainer/1.4.1
set -u

: "${VIPE_SIF_NAME:?VIPE_SIF_NAME must identify the input SIF}"
: "${VIPE_RUN_NAME:?VIPE_RUN_NAME must identify the Scratch workspace}"
SIF="${WORK}/images/${VIPE_SIF_NAME}"
WORKSPACE="${SCRATCH}/runs/vipe/${VIPE_RUN_NAME}"
INPUT_ROOT="${SCRATCH}/datasets/scannet_v2/vipe_format"
RAW_ROOT="${SCRATCH}/datasets/scannet_v2/scans"
[[ -r "${SIF}" ]] || { echo "missing image: ${SIF}" >&2; exit 1; }
rm -rf "${WORKSPACE}"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1
apptainer exec --nv --cleanenv --bind "${WORK}:${WORK},${SCRATCH}:${SCRATCH}" "${SIF}" \
    env PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 NUMEXPR_MAX_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 \
    python3 /opt/vipe/scripts/scannet_vipe_bench_evaluator.py \
    --scenes scene0001_01 scene0002_00 scene0006_02 scene0009_00 scene0009_01 scene0010_01 scene0016_01 scene0017_01 scene0020_00 scene0029_02 scene0038_02 scene0042_02 scene0046_02 scene0050_00 scene0053_00 scene0062_02 \
    --work-dir "${WORKSPACE}" \
    --input-root "${INPUT_ROOT}" \
    --raw-root "${RAW_ROOT}" \
    --do-final-eval
