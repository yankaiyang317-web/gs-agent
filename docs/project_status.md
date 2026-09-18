# Current project status

Last updated: 2026-08-18

## Working now

- Headless canonical 3DGS PLY rendering on CUDA, with RGB, accumulated
  camera-space z-depth, alpha, and a pseudocolor depth preview.
- Depth is treated as reliable only where accumulated alpha is at least `0.7`.
  Lower-alpha pixels are black in the depth preview and invalid for pixel/box
  unprojection; RGB is still returned unchanged. This is a depth-confidence
  rule, not an automatic navigation or collision block.
- Full C2W pose storage with six-direction translation and horizon-stable
  yaw/pitch navigation. Agent roll remains disabled.
- Collision-aware navigation with versioned splat-transform smooth GLB assets, Coal
  BVHs, one analytic 1.4-unit capsule, guarded recovery, and pixel or tight-box
  target approach using rendered depth.
- Coverage-first campaign v3 with automatic checkpoint topology, quantized
  spatial/heading coverage, route frontiers, regional 6DoF viewpoints,
  repetition-aware backtracking, and candidate-exhaustion guidance.
- Open connected spaces are represented by a small number of macro routes
  toward distinct directions, occluded regions, or semantic destinations.
  These candidates are generic and may coexist with narrower door, corridor,
  road, stair, gap, or aerial routes; no scene-specific "hall" rule exists.
- Video segment length and FPS are runtime parameters persisted in
  `campaign.json`. Old campaigns without those fields migrate to 81 frames and
  9 FPS, while resumed campaigns keep their recorded format.
- Missing complete clips and final partial clips use the campaign's persisted
  segment length and FPS during FFmpeg export.
- Persistent Streamable HTTP deployment allows MCP clients to reconnect without
  spawning another GPU runtime.
- A completed campaign may switch safely to another scene in the persistent
  runtime. An incomplete campaign blocks a cross-scene switch.
- Campaign implementation is now explicit: `CampaignBase` owns persistence,
  frames, video, and quality compatibility; the public `ExplorationCampaign`
  owns current coverage-first topology. The old circular v3 alias is removed.
- Exploration status includes a compact current-checkpoint/current-region
  summary (heading coverage and local candidate counts) without exposing the
  complete Campaign graph to the Agent.
- Scene manifests are deployment data; private validation scene names and assets are not enumerated in the public documentation.

## Runtime profiles

- Development defaults: 960x720, 81 frames per clip, 9 FPS, output below
  `outputs/exploration_runs`.
- Demo Campaign defaults: 2560x1920 (4:3, no crop), 270 frames per clip, 9 FPS,
  30 seconds per clip, output below `outputs/demo_runs`.
- Both profiles use the same persistent HTTP service. Profile and optional
  width/height/segment/FPS overrides are selected in
  `gs_configure_campaign`; no Demo or SSH-stdio launcher exists.

Video boundaries affect recording and export only. They do not reset the
coverage graph, candidates, topology, recent-path memory, movement speed,
translation/rotation interpolation, turning speed, or collision behavior.

## Validation status

- Windows CPU test suite: 94 tests passed on 2026-08-18.
- Real-scene CUDA, collision, and coverage validation is deployment-specific because large assets are not distributed with this repository.

## Current limitations and risks

- Exploration candidates are still proposed by the visual Agent. Candidate
  reporting converts pixels to rays but does not programmatically score their
  rendered depth/alpha or extract free-space frontiers from the depth image.
- A visible far region is not automatically proof of travel feasibility;
  mesh-capsule collision and `navigation_guard` remain the authoritative motion
  feedback.
- Batch scheduling and simultaneous multi-campaign ownership are not
  implemented. The persistent service intentionally supports one active scene
  and campaign at a time.
- Coordinate transforms are manually verified per scene. Current generated
  assets declare the verified Z-180 renderer-to-mesh transform explicitly in
  each manifest; it is not hidden in the collision backend.
- `gs_set_pose` remains a debug capability and is not collision checked.
- Shallow downward `gs_move` forward/backward collisions can use one fully
  checked horizontal or minimally diagonal-up path. This capsule-clearance
  assist never auto-descends, does not model gravity, and does not affect
  vertical movement or depth-target approaches.
- Camera capsule size and swept-motion collision step are declared per scene manifest and must be validated against each authorized collision asset.
- `world_up` is inferred from the initial image. A sideways or upside-down
  source camera therefore produces a correspondingly tilted navigation frame.

## Next validation work

Run controlled coverage comparisons from several starting cameras and scenes,
checking that narrow routes and open-space macro routes both survive candidate
deduplication without creating redundant rays. Use the results to tune generic
ranking and deduplication only; do not introduce scene-name or room-type
special cases.

## Output policy

Generated artifacts belong under `outputs/`. Historical pre-cleanup artifacts
remain under `outputs/archive/2026-08-05-pre-cleanup/`. Development and Demo
runs use separate roots and must not overwrite one another.
