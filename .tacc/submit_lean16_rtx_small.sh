#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || $1 != *@sha256:* ]]; then
    echo "usage: $0 ghcr.io/OWNER/IMAGE@sha256:DIGEST" >&2
    exit 2
fi

IMAGE_REF=$1
REMOTE_PROJECT=/work2/09672/smodak/stampede3/projects/vipe-tacc
GIT_SHORT=$(git rev-parse --short=7 HEAD)
SIF_NAME="vipe-${GIT_SHORT}-cu128.sif"
RUN_NAME="evaluation_scannet_lean16_${GIT_SHORT}"

tacc exec stampede3 -- mkdir -p "${REMOTE_PROJECT}/.tacc/logs"
tacc transfer rsync local .tacc/ stampede3 "${REMOTE_PROJECT}/.tacc/"

PULL_JSON=$(tacc jobs submit stampede3 \
    "${REMOTE_PROJECT}/.tacc/slurm/pull_image_lean16.slurm" \
    --cwd "${REMOTE_PROJECT}" \
    --export "VIPE_IMAGE_REF=${IMAGE_REF}" \
    --export "VIPE_SIF_NAME=${SIF_NAME}" \
    --json)
PULL_ID=$(printf '%s' "${PULL_JSON}" | python3 -c 'import json,sys; data=json.load(sys.stdin); assert data["ok"]; print(data["job_id"])')

RUN_JSON=$(tacc jobs submit stampede3 \
    "${REMOTE_PROJECT}/.tacc/slurm/run_lean16_rtx_small.slurm" \
    --cwd "${REMOTE_PROJECT}" \
    --afterok "${PULL_ID}" \
    --export "VIPE_SIF_NAME=${SIF_NAME}" \
    --export "VIPE_RUN_NAME=${RUN_NAME}" \
    --json)
RUN_ID=$(printf '%s' "${RUN_JSON}" | python3 -c 'import json,sys; data=json.load(sys.stdin); assert data["ok"]; print(data["job_id"])')

printf 'image:    %s\n' "${IMAGE_REF}"
printf 'pull job: %s\n' "${PULL_ID}"
printf 'run job:  %s (afterok:%s)\n' "${RUN_ID}" "${PULL_ID}"
printf 'results:  $SCRATCH/runs/vipe/%s\n' "${RUN_NAME}"
printf 'monitor:  tacc jobs show stampede3 %s\n' "${RUN_ID}"
