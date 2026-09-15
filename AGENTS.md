# gs-agent navigation defaults

Apply these instructions only when using this repository's `gs_agent` MCP
server to navigate or inspect a 3DGS scene. They do not constrain ordinary
software-development work in this repository. A user's explicit current-task
instruction takes precedence.

## Default operating behavior

- For every runtime coverage or inspection task, start with
  `gs_configure_campaign(scene="<requested>", video_clips=N, objective=...)`.
  This single call loads the server-side scene and sets the exact clip target.
  The loaded scene is fixed for that MCP task; use a new task for another scene
  and never interpret a clip boundary as permission to switch.
- Observe before consequential movement and after a meaningful view change.
- Treat the starting image-up direction as the session's stable vertical axis.
  Use world-vertical `up`/`down` movement to inspect meaningfully different
  height layers; do not remain at the initial elevation merely because defects
  are already visible there. When a coherent ground, path, courtyard, or other
  walkable-looking surface is visible below, inspect it from a safe lower
  viewpoint before continuing elsewhere. This is a coverage requirement, not
  a restriction to human-eye height: include elevated and low viewpoints when
  they expose spatially distinct reconstruction.
- Use `gs_move` for navigation-relative movement and `gs_approach_target` only for a visible
  pixel or box target. Prefer a clearly identifiable `pixel=[u,v]`: its ray and
  depth are locally consistent. Use `bbox=[x0,y0,x1,y1]` only for a tight,
  depth-consistent surface. Do not use a box spanning a doorway, opening,
  corridor, window, table, or other mixed-depth structure. Treat all distances
  as scene units.
- Treat collision and `navigation_guard` results as hard motion feedback. When
  `recovery_required=true`, do not retry the unchanged direction or target:
  observe, rotate, or choose a materially different route.
- If a navigation-action budget is explicitly configured, stop when it is
  exhausted and report what was found, what was inspected, and why the
  requested target was not confirmed. The default budget is disabled.

## Reconstruction quality is task-dependent

- For ordinary navigation, visually incomplete, blurry, floating, or apparent
  boundary regions are soft warnings. Do not repeatedly push deeper into one
  when it is unrelated to the task, but do not treat visual quality alone as a
  hard collision or proof that a route is impossible.
- For a reconstruction-defect or boundary-inspection task, those same regions
  are valid targets. Approach only to a safe nearby viewpoint, observe and
  describe them, and do not attempt to enter once collision prevents safe
  movement.

## Coverage-first free exploration

- When the task is free scene coverage rather than defect inspection, configure
  the campaign with `scene="<requested>", video_clips=N,
  objective="coverage"` before movement.
  `N` is the exact clip count for this run. Do not call `gs_record_finding`
  unless the user explicitly changes the task to reconstruction quality.
- Observe first. As soon as a credible, exploration-valued macro route is
  visible, report it with
  `gs_report_exploration_candidates` for visually supported choices only.
  Prefer following a good reported route over completing a panorama. Continue
  scanning toward a full panorama only when no credible route is visible, at a
  genuine multi-branch decision point, or when collision/backtracking requires
  an alternative. Unseen headings remain recorded on the checkpoint and may be
  scanned after returning; there is no fixed per-checkpoint yaw limit.
  `route_frontiers` are finite, exploration-valued 3D travel directions toward
  unvisited space. They may be doors, corridors, roads, stairs, gaps, open
  ground, or bounded aerial sectors; they do not require a physical path or
  surface, but must not enumerate every open ray. `viewpoint_candidates` may
  also lie in free air when moving there is expected to reveal new content or
  meaningful coarse 3D coverage. Infinite sky/void and tiny redundant pose
  changes are not candidates.
- At the first useful view of a large room, hall, courtyard, plaza, or other
  open connected area, inventory its few **macro routes** before choosing one.
  Include distinct travel axes toward a meaningful far end, a screen or large
  occluder that may hide space on either side, an open functional area such as
  a kitchen/reception/display zone, or a broad connection into another part of
  the scene. A route does not need a doorway-shaped boundary. Seeing the far
  end from the current camera does not count as spatially covering it: report
  it when translating there would create new checkpoints, reveal occluded
  space, or materially change the view. Use one candidate per distinct
  connected direction or semantic destination, not one candidate per open ray.
- Do not let a salient narrow doorway suppress a simultaneously visible
  open-area macro route. If both lead to different unvisited space, report both
  before committing; the campaign can preserve the unchosen branch for later
  backtracking. Treat the two sides of a large screen or island as separate
  candidates only when they plausibly reveal different hidden space.
- Treat intermediate poses along one coherent corridor, road, or open travel
  axis as transit checkpoints. After arriving, inspect the changed view and
  report a credible continuation when one is visible, then keep moving; do not
  rotate merely to complete unseen headings at an ordinary transit point, and
  do not invent branches from every open ray. This local inspection happens
  before acting on a suggestion to backtrack, so a saved older branch does not
  pull the Agent away before it can recognize that the current route continues.
- Treat genuine junctions as decision points: intersections, side entrances,
  stairs, halls, newly entered open areas, collision-recovery locations, and
  large occluders with plausibly distinct space on different sides. At such a
  point, report the few visually supported semantic destinations before leaving
  when practical. Follow one good route and leave the unchosen branches in the
  campaign for later graph backtracking; do not repeatedly turn between them.
- Query `gs_get_exploration_status` regularly and when motion begins to repeat.
  Prefer a local untried route, then an evidence-backed regional viewpoint,
  then physically backtrack along the suggested checkpoint path.
- Treat `route_continuation_check.due=true` as a neutral request to inspect the
  current view, not as a preference or instruction to turn back. Continue along
  a long corridor, road, coherent open area, or other meaningful structure when
  it plausibly extends ahead, including when its reconstruction quality is poor.
  Choose another saved route only after a fresh view supports that the current
  direction is becoming a scene-ending boundary. When such a boundary is useful
  coverage, approach once to a safe nearby viewpoint before ending that
  direction; do not repeatedly re-report diffuse void as a new forward route.
- Prefer semantically plausible human or drone exploration routes as a soft
  priority: roads, paths, courtyards, doors, gates, corridors, stairs, rooms,
  overlooks, and navigable open air usually outrank dense foliage, tree
  canopies, or views extremely close to the ground. Ground-level walking and
  free-air flight are both encouraged. This is a ranking preference, not a
  prohibition: vegetation, low viewpoints, and unusual routes remain valid
  when they reveal spatially distinct content or are the best remaining
  coverage opportunity.
- At one checkpoint, rotate only while doing so adds an unseen heading or is
  needed to face a reported, untried route before moving. Once its panorama is
  covered, make spatial progress or backtrack; never alternate yaw directions
  over already covered headings to fill recording frames.
- Checkpoints and graph edges are server-owned and automatic. Never use
  `gs_set_pose` to simulate exploration progress. `gs_restore_checkpoint` is
  reserved for the configured video-segment boundary or reported
  repetition/collision recovery.
- A video-segment boundary segments recording only. It does not reset topology,
  candidates, coverage regions, or recent-path memory.

## Completion

Follow direct user movement instructions when supplied, subject to collision
feedback. At the end of any navigation or inspection task, give a concise
result: success/failure, final viewpoint or evidence, and any relevant blocked
route.

For a continuous reconstruction-quality exploration request, finding one poor
region is not completion. Observe it from a safe viewpoint, then continue to
seek spatially distinct additional poor or incomplete regions. Do not prioritize
defect-category diversity: the same defect is worth recording when it appears
in another area, while a repeated view of the same area is not. The configured
video boundary is only a recording segment; continue until the configured clip
target is reached.

Recording a low-quality region alone does not make it forbidden: keep exploring
it if the scene remains semantically coherent and movement makes progress. Set
`boundary_candidate=true` on that same record only when the current forward
direction looks like a scene-ending garbage/termination region after a second
view, or after one such view plus collision/near-zero progress. This creates a
soft warning, not a block. The MCP server marks that directed route blocked only
when a later forward attempt makes near-zero collision progress; a successful
attempt clears the warning.
