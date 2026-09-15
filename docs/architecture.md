# Architecture

## Current request path

1. `gs_mcp.runtime_server` starts a scene-independent FastMCP process.
2. `SceneRuntime` discovers server-side manifests and loads one selected scene.
3. `GSplatRenderer`, `GSEnvironment`, collision, campaign, and writer form one replaceable scene session.
4. `EnvironmentToolsProxy` keeps the MCP tool registrations stable across scene switches.
5. `EnvironmentTools` implements observe, move, rotate, target approach, and campaign operations.
6. Campaign writers save frames, trajectories, and parameterized clips in one isolated run per Campaign.

A persistent Streamable HTTP deployment may listen on loopback and be reached
through a standard SSH local forward. Network routing remains separate from the
MCP and runtime implementation.

The MCP starts unloaded but advertises one stable 11-tool schema. Runtime
`gs_configure_campaign` validates the scene, exact clip target, objective,
development/Demo profile, and optional render/video overrides in one call.
An incomplete Campaign cannot be replaced. A completed Campaign is finalized
and unloaded before the same persistent process creates the next isolated run.

Each Campaign allocates a unique directory below its profile run root. The run
owns one `campaign.json`, debug frames, continuous RGB timeline, trajectory,
and videos. Development and Demo outputs remain separate without starting
different server processes.

Campaign code has two explicit layers: `CampaignBase` in `campaign_base.py`
owns persistence, frames, video, and reconstruction-quality compatibility;
the public `ExplorationCampaign` in `campaign.py` adds current coverage-first
topology. There is no version alias or circular Campaign import.

## Navigation boundary

`CameraController` derives one fixed `world_up` vector from the initial
camera's image-up direction. Public navigation uses view-forward movement,
horizon-level strafing, world-vertical ascent/descent, fixed-up yaw, and
clamped pitch. Roll remains representable in a raw C2W pose but is not an
Agent navigation action. Rotation recording samples the same stable yaw/pitch
path instead of generic quaternion interpolation, so encoded intermediate
frames do not introduce roll absent from the endpoints.

Campaign heading coverage projects view-forward onto the plane perpendicular
to the stored campaign `world_up`. Campaign v3 keeps heading coverage separate
from traversal topology: physical translations create checkpoint edges, while
rotation, observation, direct pose placement, and checkpoint restore do not.
Older campaigns are migrated in place without changing their frame timeline.

## Exploration memory boundary

The server owns checkpoint creation, spatial merging, topology edges, coarse
coverage cells, repetition detection, and persisted safe poses. The Agent can
only annotate a pixel direction at its current pose. Route frontiers are finite
travel directions toward unvisited 3D space and may include bounded free-air
sectors; they are bound to the current checkpoint. Viewpoint candidates are
bound to a coarser coverage region so an outdoor area does not create one
vertical task per checkpoint.

Height coverage is adaptive. A region starts with its observed height and gains
another sample only after an Agent-supported viewpoint produces sufficient
accumulated vertical or regional displacement. No fixed floor count is assumed.
This is structured memory and rule-based planning feedback, not an RL policy.

Collision uses a scene-configured vertical capsule for human-sized clearance
without adding gravity or rigid-body physics. Every maintained scene uses
`collision_mesh` with Coal 3.x: the GLB builds one in-memory BVH at scene load,
then each 0.02 path sample checks one analytic capsule. The standard capsule
has radius 0.25 and centerline 0.9 (1.4 total scene-unit height). There is no
voxel backend or automatic fallback: a configured mesh that cannot load fails
explicitly before navigation begins.

## Coordinate boundary

Agent poses are expressed in source PLY world coordinates. Current
splat-transform collision assets apply `(x, y, z) -> (-x, -y, z)`. Mesh scenes declare
the equivalent rigid `collision.world_to_asset` matrix explicitly in their
manifest. This matches the currently verified PlayCanvas/SuperSplat Z-180
assets without asserting that every future asset uses that convention.

## Multi-scene boundary

`SceneRuntime` remains single-active-Campaign: another scene/profile cannot
replace unfinished work. The next task calls `gs_configure_campaign` after
completion; the service frees the old renderer and CUDA state before loading
the new scene. Automated batch scheduling remains separate future work.
