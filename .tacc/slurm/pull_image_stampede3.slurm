#!/bin/bash
#SBATCH -J vipe-pull-image
#SBATCH -A IRI26004
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o .tacc/logs/%x-%j.out
#SBATCH -e .tacc/logs/%x-%j.err

set -eo pipefail
module load tacc-apptainer/1.4.1
set -u

SIF="${WORK}/images/vipe-2cf899c-cu128.sif"
PARTIAL="${SIF}.partial.sif"
mkdir -p "$(dirname "${SIF}")"
rm -f "${PARTIAL}"
apptainer pull "${PARTIAL}" docker://ghcr.io/sadanand1120/vipe-tacc@sha256:f12a0a980c6e2f49bd59f487d019c7977306c547c32fb5ceef56865510a8b211
mv "${PARTIAL}" "${SIF}"
apptainer inspect "${SIF}"
