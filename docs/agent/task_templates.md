# Agent task prompt templates

These are optional task prompts for a vision-capable Agent using the gs-agent
MCP server. They are not runtime modes and do not restrict arbitrary user
instructions. The user's current natural-language task always takes priority.

## Base operating rules

> Use `gs_observe` before consequential movement and after a meaningful change
> of view. Move with `gs_move`; forward/backward follow the view, left/right
> remain level, and up/down use the fixed session vertical axis. Use
> `gs_approach_target` only for a visible target. Treat collision and
> `navigation_guard` responses as hard
> feedback: after `recovery_required=true`, observe, rotate, or choose a
> materially different direction rather than repeating the same action. If the
> navigation budget is exhausted, stop and state what you did and did not find.
> Do not infer real-world metres from scene units.

## Navigation / find a target

> Find **[TARGET]** in this scene and move to a safe nearby viewpoint. Use RGB
> and depth preview to identify plausible routes. A blurry or incomplete-looking
> region is only a soft warning: do not keep pushing into it if it is unrelated
> to the target, but it does not by itself prove that the route is impossible.
> Prefer a clearly identifiable pixel for approach. Use `bbox` only when it is
> tight around one depth-consistent surface; do not box an entire door, entrance,
> corridor, window, or table.
> If you cannot find the target within the available exploration budget, stop
> and report the most plausible locations inspected and why they were rejected.

## Reconstruction-boundary / defect inspection

> Find and document visually incomplete or poorly reconstructed regions of this
> scene. Such regions are the target, not a reason to avoid observation. Move
> only to a safe nearby viewpoint, capture an observation, and describe the
> visible issue. Do not try to enter a region once collision prevents safe
> movement; documenting it from the boundary is sufficient.

## Continuous reconstruction-quality exploration

> Call `gs_configure_campaign(scene="[SCENE]", video_clips=N,
> objective="reconstruction_quality", profile="development")`. Freely explore this scene to find as
> many visually low-quality, incomplete,
> blurry, floating, ghosted, or apparent-boundary regions as practical. Finding
> one such region is not completion: observe it from a safe nearby viewpoint,
> then continue into spatially different areas. Do not prioritize defect-type
> diversity: record the same type again if it occurs in a new area. Do not try
> to enter a region once collision prevents safe movement. When a notable
> low-quality area is visible from a safe viewpoint, call `gs_record_finding`;
> a description is optional, and the campaign deduplicates repeat views by
> spatial location only. Do not treat a normal low-quality object or interior
> area as impassable. Set `boundary_candidate=true` only if the forward view
> appears to terminate into incoherent garbage over two views, or one such view
> plus collision/near-zero movement. That is a soft warning; only a subsequent
> near-zero collision confirms this directed route as blocked. Stop when the
> configured video-clip target is reached.
> Cover multiple useful height layers. If a coherent path, courtyard, floor,
> or other walkable-looking surface is visible below, inspect it from a safe
> lower viewpoint before spending the whole campaign at the starting height.

When the user requests N video segments, call
`gs_configure_campaign(scene="[SCENE]", video_clips=N)` before movement. `N` is the exact total
for the current automatically created run, so the frame target is
`N * video_segment_frames`.

## Coverage-first free exploration

> Call `gs_configure_campaign(scene="[SCENE]", video_clips=N, objective="coverage")`, then
> explore for broad scene coverage rather than reconstruction defects. At a
> meaningful decision point, report finite directions toward unvisited 3D
> space as `route_frontiers`; these may be doors, roads, stairs, open ground,
> gaps, or bounded aerial sectors and do not require a surface. Report a
> `viewpoint_candidate` when moving there is expected to add visual or coarse
> spatial coverage, including useful free-air views. Do not enumerate every
> open ray or pursue infinite sky/void. Query `gs_get_exploration_status` after meaningful
> progress or repetition. Prefer local untried routes, then supported regional
> viewpoints, then physically follow the suggested path back to the nearest
> unfinished checkpoint. Use `gs_restore_checkpoint` only at the configured
> video-segment boundary or for repetition/collision recovery. Do not call
> `gs_record_finding`. Continue across clip boundaries until all N clips in the
> current run are complete.
>
> In a large room, hall, courtyard, plaza, or other connected open area,
> explicitly inventory the few macro routes before selecting one. A macro route
> can be the long axis toward the far end, either side of a large screen or
> island when it hides different space, or an open kitchen, reception, display,
> or adjoining functional zone. It does not need a doorframe. Visibility from
> afar is not spatial coverage: report the route when translation would add
> checkpoints, reveal occlusion, or materially change the view. Keep one
> candidate per distinct connected direction or semantic destination, not per
> ray. If a narrow doorway and a broad open-area route lead to different
> unvisited space, report both so the unchosen branch remains in campaign
> memory.
>
> Rank routes with a light human/drone prior. Roads, paths, courtyards, doors,
> gates, corridors, stairs, rooms, overlooks, and navigable open air are often
> more useful than dense foliage, tree canopies, or extremely ground-hugging
> motion. Both walking-height and aerial exploration are encouraged. This is
> only a preference: choose vegetation, low viewpoints, or unusual free-air
> routes when they provide distinct remaining coverage. At a checkpoint, stop
> rotating once the panorama adds no unseen heading or new candidate; move or
> backtrack instead of filling frames with repeated yaw.

## Direct user-directed movement

> Follow my movement instructions exactly where they are safe. After each
> requested segment, observe and report the resulting viewpoint and any
> collision that prevented full execution.
