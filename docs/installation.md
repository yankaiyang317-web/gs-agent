# Installation

gs-agent is a normal Python project. Docker is not required unless a particular
deployment chooses to use it.

## Requirements

- Python 3.10 or newer.
- An NVIDIA GPU and compatible driver for real 3DGS rendering.
- A CUDA-enabled PyTorch build compatible with the machine.
- FFmpeg when campaign videos need to be exported.

## Create an environment

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel ninja
```

Install the CUDA-enabled PyTorch build selected for the host, then install this
repository:

```bash
pip install -e .
```

Install the development extra when tests will be run:

```bash
pip install -e ".[dev]"
```

The Linux collision backend requires Coal 3.0.3. The project dependency list is
defined in `pyproject.toml`.

## Add scene data

Large PLY files, collision GLBs, and generated runs are excluded from Git.
Obtain authorized scene assets separately and place them at the relative paths
declared by the JSON files under `scenes/`. For example:

```text
gs-agent/
|-- scenes/
|   `-- my_scene.json
`-- test_data/
    `-- my_scene/
        |-- scene.ply
        `-- scene.collision.glb
```

## Validate

Run the CPU-focused automated suite:

```bash
python -m pytest
```

Real-scene and GPU checks are separate:

```bash
python scripts/smoke/test_manifest_render.py \
  --scene-manifest scenes/my_scene.json \
  --out outputs/server_smoke
```
