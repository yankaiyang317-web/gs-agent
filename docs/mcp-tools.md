# MCP tool reference

This page documents the public tool surface exposed by gs-agent. It is a
human-readable reference; the MCP server remains the machine-readable source
of truth for tool names, parameter schemas, and Agent-visible descriptions.

## What the Agent receives automatically

After an MCP client connects, it receives every registered tool's name, input
schema, and docstring. The Agent does not need to inspect `server.py` or
`tools.py` to discover the interface. Repository guidance in `AGENTS.md` adds
task strategy when the client is operating in this repository. Harness state
checks remain authoritative even when a Prompt or Agent selects the wrong tool.

The persistent HTTP runtime adds scene discovery and lifecycle status tools.
The stdio runtime starts with one scene already selected and therefore exposes
the shorter Campaign configuration form.

## Availability and lifecycle

| Group | Tools | HTTP | stdio | Harness state |
| --- | --- | --- | --- | --- |
| Discovery | `gs_list_scenes`, `gs_describe_scene`, `gs_get_runtime_status` | Yes | No | Any |
| Task start | `gs_configure_campaign` | Yes | Yes | `idle` or `completed`; an identical active request is idempotent |
| Observation | `gs_observe`, `gs_get_pose` | Yes | Yes | `active` in HTTP mode |
| Navigation | `gs_move`, `gs_rotate`, `gs_approach_target` | Yes | Yes | `active` |
| Exploration | `gs_report_exploration_candidates`, `gs_get_exploration_status`, `gs_restore_checkpoint`, `gs_record_finding` | Yes | Yes | `active` |
| Debug | `gs_set_pose` | Yes | Yes | `active`; not collision checked |

An incompatible `gs_configure_campaign` request cannot replace an unfinished
active Campaign. Navigation calls before configuration or after completion are
rejected by the Harness.

## Discovery and task start

### `gs_list_scenes`

Lists registered server-side scene names and whether each manifest is
available. It does not load a scene or allocate renderer resources. HTTP only.

### `gs_describe_scene`

Input: `scene`.

Validates one registered manifest without loading the renderer. Returns a
safe summary including availability, camera dimensions, collision-mesh
presence, and supported runtime profiles. HTTP only.

### `gs_get_runtime_status`

Returns the Harness state (`idle`, `active`, or `completed`), current scene,
compact Campaign progress, and currently allowed tool groups. It is safe to
call first in a new conversation. HTTP only.

### `gs_configure_campaign`

Starts one user-requested Campaign.

Persistent HTTP parameters:

- `scene`: registered manifest name without `.json`.
- `video_clips`: exact positive clip total for this Campaign.
- `objective`: `coverage` or `reconstruction_quality`.
- `profile`: `development` or `demo`.
- `width`, `height`: optional paired render-size override.
- `video_segment_frames`, `video_fps`: optional recording overrides.

stdio parameters:

- `video_clips`: exact positive clip total.
- `objective`: `coverage` or `reconstruction_quality`.

Repeating the same active HTTP configuration is safe and returns the active
Campaign. A different request while a Campaign is incomplete returns
`ACTIVE_CAMPAIGN_CONFLICT` and does not switch scenes or delete outputs.

## Observation and pose

### `gs_observe`

Renders the current view. The MCP result contains:

- RGB image;
- same-resolution depth preview;
- camera-to-world matrix and camera intrinsics;
- depth and alpha shapes;
- depth-validity semantics.

Depth is accumulated camera-space z-depth. Pixels with alpha below `0.7` are
invalid for depth-based target approach.

### `gs_get_pose`

Returns the position/quaternion pose, camera-to-world matrix, and fixed
session `world_up` axis. It does not render or move the camera.

### `gs_set_pose`

Inputs: `position=[x,y,z]`, `quaternion_wxyz=[w,x,y,z]`.

Directly teleports the camera for debugging. It is intentionally not collision
checked and marks the pose as untrusted for exploration topology. Do not use it
to simulate normal progress; restore a persisted checkpoint before reporting
new exploration candidates.

## Navigation

### `gs_move`

Inputs:

- `direction`: `forward`, `backward`, `left`, `right`, `up`, or `down`.
- `distance`: finite non-negative distance in scene units.

Forward/backward are view-relative, left/right remain level, and up/down use
the fixed world vertical. The result reports requested and executed distance,
collision state, stop reason, pose, navigation guard, and Campaign progress.
Treat `recovery_required=true` as hard feedback: do not repeat the unchanged
blocked action.

### `gs_rotate`

Inputs: `yaw_deg` and `pitch_deg` (finite degrees).

Positive yaw turns right and positive pitch looks up while preserving a stable
horizon. The result includes the new pose, navigation guard, and exploration
status when a Campaign is active. Repeated already-covered rotation may be
rejected or request recovery.

### `gs_approach_target`

Supply exactly one of:

- `pixel=[u,v]`; or
- `bbox=[x0,y0,x1,y1]` for one tight, approximately single-depth surface.

Optional `stop_distance` leaves the requested distance from the reconstructed
target. The tool validates alpha/depth, performs collision-aware translation,
and returns the action metadata plus a fresh RGB/depth observation. Prefer a
single identifiable pixel; do not use a box spanning mixed depths such as a
doorway or corridor.

## Exploration and recording

### `gs_report_exploration_candidates`

Inputs are optional lists named `route_frontiers` and
`viewpoint_candidates`; at least one candidate is required. Candidate objects
identify visually supported pixels and may include a concise reason. The
runtime converts them into finite 3D directions and attaches them to the
current checkpoint. Report semantic destinations or macro routes, not every
open ray.

### `gs_get_exploration_status`

Returns compact topology coverage, current checkpoint/region information,
candidate state, repetition detection, Campaign progress, and the next
recommendation. It does not move the camera.

### `gs_restore_checkpoint`

Inputs: `checkpoint_id` and optional `reason`.

Restores a server-owned safe checkpoint only at a video-segment boundary or
when repetition/collision recovery permits it. It cannot restore after
Campaign completion. `gs_set_pose` is not a substitute for this operation.

### `gs_record_finding`

Inputs: optional `description` and `boundary_candidate` (default `false`).

Records one spatially distinct reconstruction-quality finding. Use it for the
`reconstruction_quality` objective, not ordinary coverage. Set
`boundary_candidate=true` only when repeated visual evidence or collision
suggests a scene-ending direction; this is initially a soft warning.

## Typical HTTP sequence

```text
gs_get_runtime_status()
gs_list_scenes()                         # only when discovery is needed
gs_describe_scene(scene="my_scene")
gs_configure_campaign(
    scene="my_scene",
    video_clips=2,
    objective="coverage",
)
gs_observe()
gs_report_exploration_candidates(...)
gs_move(direction="forward", distance=1.0)
gs_get_exploration_status()
```

## Keeping this reference accurate

When a public tool changes, update all three together:

1. its registration, schema, and docstring in `gs_mcp/server.py`;
2. this reference;
3. the relevant contract or lifecycle tests.

