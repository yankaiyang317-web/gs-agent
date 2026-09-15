# Coverage exploration memory

Campaign v3 separates route completeness from viewpoint richness. It is a
persistent rule-based memory layer; it does not train or run an RL policy.
The memory is persistent within one run and is not implicitly reused by the
next server start.

## Stored structures

- **Checkpoint**: an automatically merged safe pose. Nearby positions within
  `checkpoint_spacing = max(4 * camera_radius, 0.5)` reuse the same checkpoint,
  so tiny translations do not create an unbounded set of points. The Agent
  cannot create a checkpoint or choose its coordinates. Raw `gs_set_pose`
  placement is excluded from topology until a persisted checkpoint is restored.
- **Edge**: a successful physical translation between two checkpoints.
  Observation, rotation, direct pose placement, and restore never create edges.
- **Route frontier**: a finite, visually supported 3D travel direction toward
  unvisited space, stored at the current checkpoint. It can represent a door,
  road, stair, open-ground sector, the travel axis of a large interior, a
  semantic destination such as an open kitchen, or a route around a large
  occluder. It does not require a door-shaped boundary, physical path, or
  surface. Status progresses through `untried`,
  `active`, and `explored`, or becomes `blocked` after repeated collision with
  near-zero progress.
- **Coverage region**: a horizontal grouping of nearby checkpoints used to
  avoid creating a height task at every outdoor pose.
- **Viewpoint candidate**: evidence that a different 6DoF pose may add visual
  or meaningful coarse spatial coverage. The pose may lie in free air. It is
  region-scoped and becomes `valuable` only after sufficient accumulated
  height change or entry into another region.

`heading_bins` divide the horizontal panorama at each merged checkpoint into
eight 45-degree sectors. They record which directions have been seen, not
whether those directions are route exits. Coverage cells quantize translated
positions at checkpoint spacing, while coverage regions merge checkpoints over
a coarser horizontal radius (at least 1.5 scene units). These three levels keep
small pose changes from masquerading as meaningful new coverage.

## Agent loop

1. Observe and deliberately scan at a new area or decision point.
2. Report only visible route passages and evidence-backed viewpoints by pixel.
   In a connected open area, first inventory a small number of macro routes:
   distinct travel axes, semantic destinations, and occlusion-revealing sides
   of large screens or islands. A far end that is visible but has not been
   approached is still unvisited spatial coverage.
3. Query exploration status after meaningful movement or suspected repetition.
4. Prefer a local route frontier, then a regional viewpoint candidate.
5. When the local branch is exhausted, physically follow the suggested graph
   path to the nearest checkpoint with unfinished topology.
6. If candidates are exhausted, complete unseen headings at the current or
   nearest graph-connected checkpoint before concluding that no route remains.
7. Once all headings at a checkpoint are covered, another rotation over the
   same panorama is repetition; make spatial progress or backtrack.
8. Restore a saved checkpoint only at the configured video boundary or during
   recovery.

`gs_get_exploration_status` exposes global graph/candidate counts but only a
compact local detail block: the current checkpoint's seen/unseen heading bins,
scan-complete flag, and local route counts, plus the current region's coverage
cell and viewpoint counts. Full checkpoint poses and the complete graph remain
in `campaign.json` rather than being copied into every Agent response.

The campaign never enumerates directions merely because 3D space is reachable.
A large open component contributes one candidate per materially distinct route
or semantic destination, not one per ray; this keeps broad halls discoverable
without exploding the frontier count.
A coherent courtyard below, road, stair, building gap, occlusion-clearing view,
or bounded unvisited aerial sector is valid evidence. Free air is a valid
destination or transit volume; infinite sky/void and tiny redundant pose
changes are not.

## Recording boundary

Every `video_segment_frames` frames produces one MP4 clip. Topology, regions, candidates, and
recent-path memory continue across that boundary. A requested clip count is a
recording budget, not a reward and not a memory reset. A normal new process
creates a new run directory and fresh memory. Explicit `--resume-campaign` is
the only way to reopen an existing `campaign.json`.
