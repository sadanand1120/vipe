#!/bin/bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 RUN_NAME GPU_LIST MODE SCRATCH_ROOT WORK_ROOT PROJECT_DIR" >&2
    exit 2
fi

RUN_NAME=$1
GPU_LIST=$2
MODE=$3
SCRATCH_ROOT=$4
WORK_ROOT=$5
PROJECT_DIR=$6
WORKSPACE="${SCRATCH_ROOT}/runs/vipe/${RUN_NAME}"
INPUT_ROOT="${SCRATCH_ROOT}/datasets/scannet_v2/vipe_format"
RAW_ROOT="${SCRATCH_ROOT}/datasets/scannet_v2/scans"
VIPE_GIT_SHA=${VIPE_GIT_SHA:-unknown}
VIPE_IMAGE=${VIPE_IMAGE:-unknown}

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMEXPR_MAX_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

[[ -d "${INPUT_ROOT}" ]] || { echo "missing input root: ${INPUT_ROOT}" >&2; exit 1; }
[[ -d "${RAW_ROOT}" ]] || { echo "missing raw root: ${RAW_ROOT}" >&2; exit 1; }
if [[ "${MODE}" == "smoke" ]]; then
    rm -rf "${WORKSPACE}"
elif [[ "${MODE}" != "full" ]]; then
    echo "invalid mode: ${MODE}" >&2
    exit 2
fi
mkdir -p "${WORKSPACE}"

python3 - "${WORKSPACE}" "${RUN_NAME}" "${MODE}" "${VIPE_GIT_SHA}" "${VIPE_IMAGE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

workspace, run_name, mode, git_sha, image = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
metadata = {
    "status": "running",
    "run_name": run_name,
    "mode": mode,
    "git_sha": git_sha,
    "image": image,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_node": os.environ.get("SLURMD_NODENAME"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "started_at": datetime.now(timezone.utc).isoformat(),
}
(workspace / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

python3 - <<'PY'
import os
import open3d as o3d
import torch
import vipe_ext

expected_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
print(f"Python CUDA check: torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
print(f"Visible GPUs: {torch.cuda.device_count()} {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)
print(f"Torch CUDA archs: {torch.cuda.get_arch_list()}", flush=True)
print(f"Open3D CUDA available: {o3d.core.cuda.is_available()}", flush=True)
print(f"vipe_ext: {vipe_ext.__file__}", flush=True)
assert torch.__version__ == "2.7.0+cu128"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == expected_gpus
assert o3d.core.cuda.is_available()
PY

BENCH_ARGS=(
    --work-dir "${WORKSPACE}"
    --input-root "${INPUT_ROOT}"
    --raw-root "${RAW_ROOT}"
    --do-final-eval
)
if [[ "${MODE}" == "smoke" ]]; then
    BENCH_ARGS=(--scenes scene0013_01 "${BENCH_ARGS[@]}")
fi

python3 /opt/vipe/scripts/scannet_vipe_bench_evaluator.py "${BENCH_ARGS[@]}"
python3 /opt/vipe/.tacc/validate_eval.py \
    --workspace "${WORKSPACE}" \
    --mode "${MODE}" \
    --write-success

python3 - "${WORKSPACE}/run_metadata.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads(path.read_text(encoding="utf-8"))
metadata["status"] = "complete"
metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

if [[ "${MODE}" == "smoke" ]]; then
    cp "${WORKSPACE}/validation.json" "${PROJECT_DIR}/.tacc/logs/${RUN_NAME}_validation.json"
    rm -rf "${WORKSPACE}"
fi
