#!/bin/bash
#SBATCH -J vipe-full-tacc-h100
#SBATCH -A IRI26004
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o .tacc/logs/%x-%j.out
#SBATCH -e .tacc/logs/%x-%j.err

set -eo pipefail
module load tacc-apptainer/1.4.1
set -u

PROJECT_DIR="${WORK}/projects/vipe-tacc"
SIF="${WORK}/images/vipe-2cf899c-cu128.sif"
[[ -r "${SIF}" ]] || { echo "missing image: ${SIF}" >&2; exit 1; }

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID}"
export APPTAINERENV_SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION}"
export APPTAINERENV_SLURMD_NODENAME="${SLURMD_NODENAME}"
apptainer exec --nv --cleanenv --bind "${WORK}:${WORK},${SCRATCH}:${SCRATCH}" "${SIF}" \
    bash /opt/vipe/.tacc/run_eval.sh \
    evaluation_scannet_default_full_tacc 0,1,2,3 full "${SCRATCH}" "${WORK}" "${PROJECT_DIR}"
