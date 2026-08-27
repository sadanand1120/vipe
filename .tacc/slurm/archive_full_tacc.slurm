#!/bin/bash
#SBATCH -J vipe-archive-full-tacc
#SBATCH -A IRI26004
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o .tacc/logs/%x-%j.out
#SBATCH -e .tacc/logs/%x-%j.err

set -euo pipefail
PROJECT_DIR="${WORK}/projects/vipe-tacc"
bash "${PROJECT_DIR}/.tacc/archive_eval.sh" \
    "${SCRATCH}/runs/vipe/evaluation_scannet_default_full_tacc" \
    "${WORK}/projects/vipe-evaluations/evaluation_scannet_default_full_tacc" \
    "${PROJECT_DIR}"
