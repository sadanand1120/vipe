#!/bin/bash
#SBATCH -J vipe-pull-lean16
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

: "${VIPE_IMAGE_REF:?VIPE_IMAGE_REF must be a digest-addressed GHCR image}"
: "${VIPE_SIF_NAME:?VIPE_SIF_NAME must identify the output SIF}"
SIF="${WORK}/images/${VIPE_SIF_NAME}"
PARTIAL="${SIF}.partial.sif"
mkdir -p "$(dirname "${SIF}")"
rm -f "${PARTIAL}"
apptainer pull "${PARTIAL}" "docker://${VIPE_IMAGE_REF}"
mv "${PARTIAL}" "${SIF}"
apptainer inspect "${SIF}"
