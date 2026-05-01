## Build, Test, and Development Commands
- Run all commands associated to this repo inside container named 'humble' and inside a dedicated conda (/opt/miniconda3) env called 'vipe-manual'. Use docker exec to run commands inside the container.
- Edit files directly on the host repo; use the container only for executing code, tests, and tooling against the bind-mounted workspace.
- Do not delete the built `vipe_ext` extension artifact as generic cleanup. This repo imports `vipe_ext` at runtime; if it is missing, rebuild it with `pip3 install --no-build-isolation -e .` inside `vipe-manual` instead of removing it.
