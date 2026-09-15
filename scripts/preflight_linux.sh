#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SCENE_MANIFEST="${1:-${GS_SCENE_MANIFEST:-scenes/guju.json}}"
if [[ "$SCENE_MANIFEST" != /* ]]; then
  SCENE_MANIFEST="$PROJECT_ROOT/$SCENE_MANIFEST"
fi

if [[ -n "${GS_PYTHON:-}" ]]; then
  PYTHON="$GS_PYTHON"
else
  VENV_ROOT="${GS_VENV:-$PROJECT_ROOT/.venv}"
  PYTHON="$VENV_ROOT/bin/python"
fi

printf 'project_root=%s\n' "$PROJECT_ROOT"
printf 'python=%s\n' "$PYTHON"
printf 'manifest=%s\n' "$SCENE_MANIFEST"

[[ -x "$PYTHON" ]] || { printf 'Missing Python environment: %s\n' "$PYTHON" >&2; exit 2; }
[[ -f "$SCENE_MANIFEST" ]] || { printf 'Missing manifest: %s\n' "$SCENE_MANIFEST" >&2; exit 2; }
command -v ffmpeg >/dev/null 2>&1 || { printf 'ffmpeg is not installed\n' >&2; exit 2; }
ffmpeg -hide_banner -encoders 2>/dev/null | grep 'libx264' >/dev/null || { printf 'ffmpeg lacks libx264\n' >&2; exit 2; }

cd -- "$PROJECT_ROOT"
"$PYTHON" -c "import sys; assert sys.version_info >= (3, 10), sys.version; print('python_version=' + sys.version.split()[0])"
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch=' + torch.__version__); print('gpu=' + torch.cuda.get_device_name(0)); print('cuda=' + str(torch.version.cuda))"
"$PYTHON" -c "import gsplat, mcp, numpy, PIL, plyfile; print('python_imports=ok')"
"$PYTHON" -c "import sys; from gs_env import load_scene_manifest; m=load_scene_manifest(sys.argv[1]); print('scene=' + m.name); print('ply=' + str(m.ply_path)); print('collision_mesh=' + str(m.collision_mesh_path))" "$SCENE_MANIFEST"

mkdir -p -- "$PROJECT_ROOT/outputs/exploration_runs"
probe="$PROJECT_ROOT/outputs/exploration_runs/.preflight-write-probe"
: > "$probe"
rm -f -- "$probe"

printf 'preflight=ok\n'
printf 'Run the real GPU render smoke test next:\n'
printf '  %s scripts/smoke/test_manifest_render.py --scene-manifest %q --out outputs/server_smoke\n' "$PYTHON" "$SCENE_MANIFEST"
