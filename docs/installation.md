# Installation

gs-agent is a normal Python project. Docker is not required unless a particular
deployment chooses to use it.

## Requirements

- Python 3.10 or newer.
- An NVIDIA GPU and compatible driver for real 3DGS rendering.
- A CUDA-enabled PyTorch build compatible with the machine.
- FFmpeg when campaign videos need to be exported.

Real-scene GPU rendering and collision checking are currently verified on
Linux. Windows environment setup and MCP configuration are documented, but the
complete real-scene runtime has not been verified there.

## Tested environment

One verified configuration is Ubuntu 24.04, Python 3.11.15, an NVIDIA L40 with
driver 590.44.01, PyTorch 2.5.1+cu121, CUDA Toolkit 12.0, and gsplat 1.5.3.
This is a reference configuration, not a strict requirement.

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

If `gsplat` builds CUDA extensions locally, a compatible CUDA Toolkit (`nvcc`)
and C/C++ compiler are also required.

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
