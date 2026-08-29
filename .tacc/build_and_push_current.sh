#!/bin/bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
IMAGE_REPOSITORY=ghcr.io/sadanand1120/vipe-tacc
IMAGE="${IMAGE_REPOSITORY}:${GIT_SHA:0:7}"

cd "${REPO_ROOT}"
/usr/bin/docker buildx build \
    --push \
    --platform linux/amd64 \
    --progress plain \
    --file .tacc/docker/Dockerfile.tacc \
    --build-arg "VIPE_GIT_SHA=${GIT_SHA}" \
    --tag "${IMAGE}" \
    . >&2

DIGEST=$(/usr/bin/docker buildx imagetools inspect "${IMAGE}" --format '{{json .Manifest.Digest}}' | tr -d '"')
[[ "${DIGEST}" == sha256:* ]] || { echo "failed to resolve pushed image digest" >&2; exit 1; }
printf '%s@%s\n' "${IMAGE_REPOSITORY}" "${DIGEST}"
