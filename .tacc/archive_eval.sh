#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 SOURCE_DIR DEST_DIR PROJECT_DIR" >&2
    exit 2
fi

SOURCE_DIR=$1
DEST_DIR=$2
PROJECT_DIR=$3
PARTIAL_DIR="$(dirname "${DEST_DIR}")/.$(basename "${DEST_DIR}").partial"

[[ -f "${SOURCE_DIR}/_SUCCESS.json" ]] || { echo "source is not validated: ${SOURCE_DIR}" >&2; exit 1; }
[[ ! -e "${DEST_DIR}" ]] || { echo "destination already exists: ${DEST_DIR}" >&2; exit 1; }
mkdir -p "$(dirname "${DEST_DIR}")" "${PARTIAL_DIR}"

rsync -a --delete --partial --info=stats2 "${SOURCE_DIR}/" "${PARTIAL_DIR}/"
DIFF_FILE=$(mktemp "${TMPDIR:-/tmp}/vipe-archive-diff.XXXXXX")
trap 'rm -f "${DIFF_FILE}"' EXIT
rsync -ani --delete "${SOURCE_DIR}/" "${PARTIAL_DIR}/" > "${DIFF_FILE}"
if [[ -s "${DIFF_FILE}" ]]; then
    echo "archive verification found remaining rsync changes:" >&2
    cat "${DIFF_FILE}" >&2
    exit 1
fi

python3 "${PROJECT_DIR}/.tacc/validate_eval.py" \
    --workspace "${PARTIAL_DIR}" \
    --mode full \
    --reference "${PROJECT_DIR}/.tacc/full13_reference.json" \
    --write-success

python3 - "${PARTIAL_DIR}" "${SOURCE_DIR}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

destination, source = Path(sys.argv[1]), sys.argv[2]
payload = {
    "source": source,
    "archived_at": datetime.now(timezone.utc).isoformat(),
    "archive_job_id": os.environ.get("SLURM_JOB_ID"),
    "git_sha": "2cf899c83e7018c1257733537d80ceb2bceb3a72",
}
(destination / "_ARCHIVE_COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

mv "${PARTIAL_DIR}" "${DEST_DIR}"
echo "Archived ${SOURCE_DIR} -> ${DEST_DIR}"
