# Scripts

gs-agent is installed and run as a normal Python project. Scripts in this
directory validate an installation, exercise real scenes, or provide
development utilities; they are not required package entrypoints.

## Environment preflight

`preflight_linux.sh` checks:

- Python 3.10 or newer.
- CUDA-enabled PyTorch and GPU availability.
- Required Python imports.
- FFmpeg and the libx264 encoder.
- Scene manifest and asset path resolution.
- Output directory permissions.

```bash
bash scripts/preflight_linux.sh scenes/guju.json
```

## Real-scene and GPU smoke checks

`smoke/` contains focused validation scripts for manifest rendering, camera
actions, target approach, COLMAP initial-view conversion, and explicit poses.
They may require a CUDA GPU and real scene assets and commonly write images to
`outputs/` for manual inspection.

These checks answer questions such as "does this scene load and render upright?"
They do not start or supervise a production service.

## Development tools

`tools/` contains standalone utilities:

- `benchmark_renderer.py` measures renderer load and frame performance.
- `rank_initial_poses.py` scores candidate starting viewpoints.
- `validate_collision_mesh.py` inspects collision meshes and probe results.
- `export_debug_video.py` exports recorded frames as videos.

## Collision asset generation

`generate_collision_mesh.sh` wraps the external splat-transform tool to filter
