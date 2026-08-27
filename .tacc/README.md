# ViPE ScanNet TACC workflow

This workflow packages Git commit `2cf899c83e7018c1257733537d80ceb2bceb3a72` and runs the full13-equivalent ScanNet evaluation on Stampede3.

BuildKit pushes directly because this host's Docker endpoint is backed by Podman and cannot load this large image reliably. The pushed linux/amd64 manifest is `sha256:f12a0a980c6e2f49bd59f487d019c7977306c547c32fb5ceef56865510a8b211`.

```bash
/usr/bin/docker buildx build --push --platform linux/amd64 -f .tacc/docker/Dockerfile.tacc -t ghcr.io/sadanand1120/vipe-tacc:2cf899c .
tacc exec stampede3 -- mkdir -p /work2/09672/smodak/stampede3/projects/vipe-tacc/.tacc/logs
tacc transfer rsync copy local .tacc/ stampede3 /work2/09672/smodak/stampede3/projects/vipe-tacc/.tacc/
tacc jobs submit stampede3 '$WORK/projects/vipe-tacc/.tacc/slurm/pull_image_stampede3.slurm' --cwd '$WORK/projects/vipe-tacc'
```

Only `.tacc/` is synchronized. The executable repository is baked into the image, and avoiding a generic repository sync prevents local benchmark workspaces from being transferred.

After the image pull finishes, submit both queue-specific smoke jobs. Submit the two production jobs only after both smoke validations pass.

```bash
tacc jobs submit stampede3 '$WORK/projects/vipe-tacc/.tacc/slurm/smoke_h100.slurm' --cwd '$WORK/projects/vipe-tacc'
tacc jobs submit stampede3 '$WORK/projects/vipe-tacc/.tacc/slurm/smoke_rtx_small.slurm' --cwd '$WORK/projects/vipe-tacc'
tacc jobs submit stampede3 '$WORK/projects/vipe-tacc/.tacc/slurm/run_full_tacc_h100.slurm' --cwd '$WORK/projects/vipe-tacc'
tacc jobs submit stampede3 '$WORK/projects/vipe-tacc/.tacc/slurm/run_full_tacc2_rtx_small.slurm' --cwd '$WORK/projects/vipe-tacc'
```

Queue each archive after its matching production job succeeds:

```bash
sbatch --dependency=afterok:<H100_JOB_ID> "$WORK/projects/vipe-tacc/.tacc/slurm/archive_full_tacc.slurm"
sbatch --dependency=afterok:<RTX_JOB_ID> "$WORK/projects/vipe-tacc/.tacc/slurm/archive_full_tacc2.slurm"
```

The Scratch workspaces remain in place. The archive jobs copy them atomically to distinct Work directories:

- `$WORK/projects/vipe-evaluations/evaluation_scannet_default_full_tacc`
- `$WORK/projects/vipe-evaluations/evaluation_scannet_default_full_tacc2`
