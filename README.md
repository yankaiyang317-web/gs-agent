# gs-agent

`gs-agent` exposes a 3D Gaussian Splatting scene to a vision-capable Agent
through the Model Context Protocol (MCP). It provides RGB/depth observations,
horizon-stable 6DoF navigation, mesh-capsule collision checks, visible-target
approach, coverage memory, campaign persistence, and video recording.

The project is installed as a normal Python environment. Docker is not required.

## Capabilities

- Render RGB, accumulated camera-space depth, alpha, and depth previews.
- Move forward/backward, strafe, ascend/descend, yaw, and pitch.
- Keep a stable session vertical axis and disable Agent-controlled roll.
- Reject unsafe paths using a swept analytic capsule and a collision GLB.
- Approach a visible pixel or a tight, depth-consistent bounding box.
- Explore for broad coverage with checkpoints, route candidates, and backtracking.
- Find a user-requested semantic target and move to a safe nearby viewpoint.
- Persist Campaign state, observations, trajectories, and video clips.
- Run as a local stdio MCP process or a persistent Streamable HTTP service.

## Agent system: Prompt, Skill, and Harness

| Concept | Current status | Documentation |
| --- | --- | --- |
| Prompt | Implemented through `AGENTS.md`, task templates, and MCP tool descriptions. | [Prompt system](docs/prompt-system.md) |
| Skill | Reusable workflows exist, but no formal `SKILL.md` package is enabled yet. | [Skill status](docs/skill-system.md) |
| Harness | Explicit `GSAgentHarness` lifecycle boundary over the runtime, Campaign, tools, and persistence. | [Harness architecture](docs/harness.md) |

The model loop is hosted by an MCP client such as Codex. gs-agent provides the
environment-side Harness: it exposes observations and actions, validates tool
calls, applies collision and navigation guards, maintains exploration state,
and records reproducible outputs.

## 1. Requirements

For real 3DGS scenes:

- Python 3.10 or newer.
- An NVIDIA GPU and compatible driver.
- A CUDA-enabled PyTorch build compatible with that driver.
- `gsplat==1.5.3`.
- `coal==3.0.3` on Linux for collision checking.
- FFmpeg with `libx264` when exporting MP4 files.
- A Gaussian Splatting PLY file and a matching static collision GLB per scene.

The repository does not include the multi-gigabyte real scene assets. It only
contains `tests/fixtures/smoke_scene.ply`, a tiny fixture for automated tests.

## 2. Install the Python environment

Linux:

```bash
git clone https://github.com/yankaiyang317-web/gs-agent.git
cd gs-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja
```

Windows PowerShell:

```powershell
git clone https://github.com/yankaiyang317-web/gs-agent.git
cd gs-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel ninja
```

Install the CUDA-enabled PyTorch build appropriate for the new machine first.
Do not copy a PyTorch command from another host without checking its driver and
CUDA compatibility. Then install gs-agent:

```bash
pip install -e .
```

See [installation](docs/installation.md) for validation notes.

## 3. Provide scene data

A runnable scene needs two aligned assets:

1. A canonical 3DGS `.ply` containing the Gaussian properties required by gsplat.
2. A static triangle `.collision.glb` aligned to the PLY coordinate system.

Keep large assets outside Git but place or link them at the paths declared by
the scene manifest. The maintained layout is:

```text
gs-agent/
|-- scenes/
|   |-- guju.json
|   `-- ...
|-- test_data/                 # ignored by Git
|   |-- guju/
|   |   |-- 1lpn0524.ply
|   |   `-- 1lpn0524_v006.collision.glb
|   `-- ...
`-- outputs/                  # ignored by Git
```

The included manifests expect these files:

| Scene | Gaussian PLY | Collision GLB |
| --- | --- | --- |
| `bangongshi` | `test_data/bangongshi/65b88b72a3604a2180248f1f666d1b30.ply` | `test_data/bangongshi/65b88b72a3604a2180248f1f666d1b30_v004.collision.glb` |
| `changguan` | `test_data/changguan/changguan.ply` | `test_data/changguan/changguan_v060.collision.glb` |
| `guju` | `test_data/guju/1lpn0524.ply` | `test_data/guju/1lpn0524_v006.collision.glb` |
| `jiudian` | `test_data/jiudian/y253p58x.ply` | `test_data/jiudian/y253p58x.collision.glb` |
| `laojie` | `test_data/laojie/nl039zwl.ply` | `test_data/laojie/nl039zwl_v006.collision.glb` |
| `tiyuchang` | `test_data/tiyuchang/35162acf9e5143dbaa997e8501057520.ply` | `test_data/tiyuchang/35162acf9e5143dbaa997e8501057520_v060_top_unbounded.collision.glb` |

On another authorized machine, the current private assets can be copied from a
data host without adding them to Git:

```bash
rsync -av --progress render-host:/path/to/gs-agent/test_data/ ./test_data/
```

## 4. Define and initialize a scene

Scene-specific data is isolated in a JSON manifest under `scenes/`. Asset
paths are resolved relative to the manifest file.

Minimal maintained schema:

```json
{
  "name": "my_scene",
  "ply": "../test_data/my_scene/scene.ply",
  "collision_mesh": "../test_data/my_scene/scene.collision.glb",
  "initial_pose": {
    "position": [0.0, 0.0, 0.0],
    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
  },
  "world_up": [0.0, -1.0, 0.0],
  "camera": {
    "width": 960,
    "height": 720,
    "fx": 480.0,
    "fy": 480.0,
    "cx": 480.0,
    "cy": 360.0
  },
  "collision": {
    "camera_radius": 0.25,
    "camera_body_height": 0.9,
    "step": 0.02,
    "world_to_asset": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]
  }
}
```

The pose is camera-to-world in the PLY coordinate system and the quaternion is
ordered `[w, x, y, z]`. The PLY and collision mesh must use the transform
declared by `collision.world_to_asset`. Do not guess the initial pose or
vertical direction: render it and visually confirm that the image is upright
and the initial capsule is collision-free.

Validate path resolution without rendering:

```bash
python -c "from gs_env import load_scene_manifest; print(load_scene_manifest('scenes/guju.json'))"
```

On Linux, run the environment preflight:

```bash
bash scripts/preflight_linux.sh scenes/guju.json
```

Then perform one real GPU render:

```bash
python scripts/smoke/test_manifest_render.py \
  --scene-manifest scenes/guju.json \
  --out outputs/server_smoke
```

Inspect `outputs/server_smoke/rgb.png` before using the scene with an Agent.
See [scene manifests](docs/scene-manifests.md) for coordinate and collision
details.

## 5. Connect through MCP

### Option A: local stdio

Use stdio when the MCP client and renderer run on the same machine. Copy
`.codex/config.stdio.example.toml` into the relevant client configuration and
replace the absolute paths:

```toml
[mcp_servers.gs_agent]
command = "/absolute/path/to/gs-agent/.venv/bin/python"
args = [
  "-m",
  "gs_mcp.server",
  "--scene-manifest",
  "/absolute/path/to/gs-agent/scenes/guju.json",
  "--exploration-campaign",
  "/absolute/path/to/gs-agent/outputs/exploration_runs",
]
startup_timeout_sec = 300
```

The MCP client starts the process. The scene is already selected, so Campaign
configuration uses the shorter call:

```text
gs_configure_campaign(video_clips=1, objective="coverage")
```

### Option B: persistent Streamable HTTP

Use HTTP when the renderer should remain alive independently of the client or
runs on another host:

```bash
python -m gs_mcp.runtime_server \
  --scenes-root ./scenes \
  --runs-root ./outputs/exploration_runs \
  --demo-runs-root ./outputs/demo_runs \
  --device cuda \
  --host 127.0.0.1 \
  --port 18913
```

Configure the MCP client:

```toml
[mcp_servers.gs_agent]
url = "http://127.0.0.1:18913/mcp"
startup_timeout_sec = 300
```

For a remote loopback-only server, route the connection separately:

```bash
ssh -N -L 18913:127.0.0.1:18913 user@render-host
```

The MCP configuration remains unchanged. Start an HTTP Campaign with:

```text
gs_configure_campaign(
    scene="guju",
    video_clips=1,
    objective="coverage",
    profile="development",
)
```

See the [generic MCP connection guide](docs/mcp-connection.md).
The persistent server exposes read-only discovery/status tools separately from task execution. Navigation calls are accepted only while the Harness is `active`; an identical configure request is idempotent, while an incompatible unfinished Campaign is rejected without replacing or deleting it.


## 6. Run an Agent task

The two principal task families are:

- **Coverage-first exploration:** explore broad spatial regions, preserve route
  candidates, and use Campaign memory for backtracking.
- **Target finding:** search for a requested object or place, then approach a
  visible target to a safe stopping distance.

Repository defaults are in `AGENTS.md`. Optional task prompts are in
`docs/agent/task_templates.md`. The runtime exposes tools for observation,
pose inspection, collision-aware movement, target approach, candidate
reporting, exploration status, checkpoint restoration, and Campaign setup.

A typical coverage task begins with:

```text
1. gs_configure_campaign(...)
2. gs_observe()
3. gs_report_exploration_candidates(...)
4. gs_move(...) or gs_rotate(...)
5. gs_get_exploration_status()
```

The Agent continues until the configured clip target is complete or a stated
navigation budget is exhausted.

## 7. Outputs

Each Campaign receives an isolated run directory:

```text
outputs/
|-- exploration_runs/<scene>/run_<timestamp>/
|   |-- campaign.json
|   |-- trajectory.jsonl
|   |-- debug/
|   `-- video/
`-- demo_runs/<scene>/run_<timestamp>/
```

The development profile defaults to 960x720, 81 frames per clip, and 9 FPS.
The demo profile defaults to 2560x1920, 270 frames per clip, and 9 FPS. Runtime
overrides are persisted in `campaign.json`.

## 8. Tests and developer utilities

Run the CPU-focused suite:

```bash
python -m pytest
```

- `tests/` contains automated logic and contract tests.
- `scripts/smoke/` contains real-scene/GPU checks that produce evidence for
  manual inspection; these are not service entrypoints.
- `scripts/tools/` contains renderer benchmarks, initial-pose ranking,
  collision-mesh validation, and debug-video export utilities.
- `scripts/preflight_linux.sh` checks CUDA, imports, FFmpeg, assets, and output
  permissions before a real-scene run.

## Repository layout

- `gs_env/` - renderer, camera, geometry, scene manifests, and collision.
- `gs_mcp/` - MCP tools, runtime, Campaign state, recording, and video export.
- `scenes/` - versioned scene manifests without large scene assets.
- `tests/` - automated tests and one tiny committed PLY fixture.
- `scripts/` - preflight, smoke checks, and developer utilities.
- `docs/` - architecture, navigation, Agent integration, and deployment docs.
- `.codex/` - generic MCP client configuration examples.
- `test_data/`, `outputs/` - local data and generated files ignored by Git.

## Documentation

Start with the [documentation map](docs/README.md):

- [Installation](docs/installation.md)
- [MCP connection](docs/mcp-connection.md)
- [Architecture](docs/architecture.md)
- [Navigation contract](docs/navigation-controls.md)
- [Exploration memory](docs/exploration-memory.md)
- [Agent integration](docs/agent/README.md)

## Versioning

`v0.1.0` is the first Git baseline. Earlier ZIP files are pre-Git snapshots
and may later be imported as explicitly reconstructed history.
