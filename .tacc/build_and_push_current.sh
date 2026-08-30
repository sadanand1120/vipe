#!/bin/bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
IMAGE_REPOSITORY=ghcr.io/sadanand1120/vipe-tacc
IMAGE="${IMAGE_REPOSITORY}:${IMAGE_TAG:-${GIT_SHA:0:7}}"
CACHE_IMAGE="${IMAGE_REPOSITORY}:buildcache"
CACHE_FROM=()
if [[ "${USE_CACHE_FROM:-1}" == 1 ]]; then
    CACHE_FROM=(--cache-from "type=registry,ref=${CACHE_IMAGE}")
fi

cd "${REPO_ROOT}"
/usr/bin/docker buildx build \
    --push \
    --platform linux/amd64 \
    --progress plain \
    --file .tacc/docker/Dockerfile.tacc \
    --build-arg "VIPE_GIT_SHA=${GIT_SHA}" \
    --tag "${IMAGE}" \
    "${CACHE_FROM[@]}" \
    --cache-to "type=registry,ref=${CACHE_IMAGE},mode=max,image-manifest=true,oci-mediatypes=true" \
    . >&2

DIGEST=$(/usr/bin/docker buildx imagetools inspect "${IMAGE}" --format '{{json .Manifest.Digest}}' | tr -d '"')
[[ "${DIGEST}" == sha256:* ]] || { echo "failed to resolve pushed image digest" >&2; exit 1; }
printf '%s@%s\n' "${IMAGE_REPOSITORY}" "${DIGEST}"
