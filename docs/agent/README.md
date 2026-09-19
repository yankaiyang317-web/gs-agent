# Agent integration guide

Use this document when giving a vision-capable Agent access to the gs-agent MCP
server. It is intentionally concise and contains no development roadmap.

The repository's root `AGENTS.md` supplies these default navigation behaviors
to Codex tasks using this project. The templates in this directory remain
optional task-specific additions; explicit user instructions always override
the defaults.

## Available interaction

- `gs_observe` returns a pixel-aligned RGB image and depth preview.
- `gs_move(direction, distance)` performs collision-aware navigation movement:
  forward/backward follow the view, left/right remain level, and up/down follow
  the fixed session vertical axis. A near-zero shallow downward forward/backward
  collision may be replaced by a fully collision-checked horizontal or minimal
  diagonal-up clearance path; the response explicitly reports that adjustment.
- `gs_rotate(yaw_deg, pitch_deg)` keeps a stable horizon: yaw turns around the
  fixed up direction inferred from the initial image, while pitch looks up or
  down and is clamped before inversion.
- `gs_approach_target(pixel, stop_distance, bbox=None)` approaches one visible
  target while preserving the requested stop distance.
- `gs_record_finding(description="", boundary_candidate=False)` records one
  spatially distinct low-quality region. `boundary_candidate` is only for a
  visually likely scene-ending forward direction; it is a soft route warning,
  not a ban on low-quality regions.
- On the persistent HTTP service,
  `gs_configure_campaign(scene, video_clips, objective="coverage",
  profile="development", ...)` creates the run and selects its scene, profile,
  exact clip target, and optional render/video overrides. The retained local
  Windows server uses the shorter already-loaded-scene form.
- `gs_report_exploration_candidates(route_frontiers, viewpoint_candidates)`
  reports visually supported directions by image pixel. Route frontiers are
  finite directions toward unvisited 3D space, including roads, stairs, open
  ground, and bounded aerial sectors. Viewpoint candidates may be free-air
  poses when they add expected visual or coarse spatial coverage.
- `gs_get_exploration_status()` returns global topology counts, a compact
  current-checkpoint/current-region summary (heading and local candidate
  coverage), repetition, and a recommended local action or backtrack path.
  Its `campaign` object is the authoritative completion receipt: it includes
  frame and clip progress, `complete`, the Campaign directory, and
  `completed_video_paths` for MP4 files that are present and non-empty.
- `gs_restore_checkpoint(checkpoint_id, reason="")` restores only a persisted
  checkpoint, and only at a clip boundary or during repetition/collision recovery.
- `gs_get_pose` reads pose; `gs_set_pose` is debug/MVP-only direct placement.

`gs_get_pose` also returns `world_up`. Roll is not a `gs_rotate` parameter.
Changing pose through `gs_set_pose` deliberately preserves the existing
session vertical axis. The server normally derives this axis from initial
image-up; a manifest `world_up` or CLI `--world-up X Y Z` can override it.

Scene distances are scene units, not assumed metres. RGB and depth preview are
useful visual evidence. Depth pixels with accumulated alpha below 0.7 are shown
as black and rejected by pixel/box depth targeting; this marks insufficient
depth confidence, not an automatic navigation block or a complete measure of
reconstruction quality.

## Safe behavior

Observe before consequential movement and after a meaningful view change. The
`navigation_guard` returned by movement actions is hard feedback: after
`recovery_required=true`, choose a materially different route. When the action
budget is exhausted, stop and report what was found or not found.

For ready-to-copy natural-language task prompts, use
[task templates](task_templates.md). User instructions always override those
optional templates.

For continuous coverage, do not assume a fixed number of height layers. Scan
up and down at a new region, and pursue a vertical or free-air viewpoint when
it is expected to expose new content or cover a meaningfully new bounded 3D
area. Height samples are discovered and merged per coverage region. Open air
is valid exploration space, but infinite sky/void is not an enumerable frontier.

Prefer a clearly identifiable pixel for target approach. The pixel method uses
a locally consistent ray and depth. Use a bounding box only when it tightly
contains one approximately depth-consistent surface. Do not box an entire door,
opening, corridor, window, table, or counter: mixed foreground/background depth
can place the target at an incorrect 3D point. Navigate through such structures
with short `gs_move` steps after observing.

## Per-run exploration campaign

The persistent HTTP service creates a unique
`run_YYYYMMDD_HHMMSS_microseconds` directory for every successful configure
call. No process restart or per-task config file is required. Each run contains
its own `campaign.json`, debug observations, RGB timeline, trajectory, and
`video/clip_000.mp4` onward. The local Windows entry may still use
`--exploration-campaign ROOT` directly.

`campaign.json` is runtime state, not scene configuration. It stores this run's
frame count, exact clip target, checkpoints, graph edges, candidates, and
coverage. Processes never share it. Use `--resume-campaign RUN_PATH` only when
explicitly resuming an interrupted run; normal startup always begins from the
scene manifest's initial pose with fresh memory.

`video_clips=N` means exactly N clips for the current run. The target is
`N * video_segment_frames`. The Campaign keeps topology and coverage continuous
between clips, but not between independent runs.
