## Build, Test, and Development Commands
- Run all commands associated to this repo inside container named 'humble' and inside a dedicated conda (/opt/miniconda3) env called 'vipe-manual'. Use docker exec to run commands inside the container.
- Edit/view files directly on the host repo; use the container only for executing code, tests, and tooling against the bind-mounted workspace.
- Do not delete the built `vipe_ext` extension artifact as generic cleanup. This repo imports `vipe_ext` at runtime; if it is missing, rebuild it with `pip3 install --no-build-isolation -e .` inside `vipe-manual` instead of removing it.

## Local Codex Memory
- ScanNet eval dashboard: use `scripts/scannet_eval_dashboard.py` to compare two eval workspaces, e.g. `python3 scripts/scannet_eval_dashboard.py --before-root workspace/evaluation_scannet_default_full --after-root workspace/evaluation_scannet_default_full5 --input-root data/scannet --host 127.0.0.1 --port 18769`.
- To expose the dashboard publicly, use Cloudflare's `cloudflared` binary. If missing, download it with `wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared`. Launch `./cloudflared tunnel --url http://127.0.0.1:<port> --protocol http2 --no-autoupdate` in a persistent exec session and return the printed `trycloudflare.com` URL.
- Do not background the dashboard/tunnel with plain `&`/`nohup` in this environment; those jobs got reaped. Use persistent `exec_command(..., tty=true)` sessions for both.
- To stop dashboard views, kill the specific PIDs from `pgrep -af 'scannet_eval_dashboard|cloudflared tunnel'`.
