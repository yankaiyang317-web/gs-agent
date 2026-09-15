# Navigation controls

This document defines the current Agent-facing camera and movement contract.
The renderer still consumes a full camera-to-world (C2W) pose, but ordinary
navigation deliberately excludes camera-axis roll.

## Coordinate convention and session vertical

Camera coordinates follow OpenCV/gsplat:

- `+X`: image right
- `+Y`: image down
- `+Z`: view forward

When `GSEnvironment` is created, it derives a fixed vertical direction from the
initial camera pose:

```python
world_up = -initial_camera_to_world[:3, 1]
```

This fallback primarily removes initial roll. A single camera pose cannot
recover physical gravity: if the input image is strongly pitched upward or
downward, its image-up vector is also tilted relative to gravity. When a more
reliable scene axis is known, supply `world_up` in the scene manifest or pass
`--world-up X Y Z`; the explicit value takes precedence and is normalized.

For a COLMAP input image, pose conversion and any COLMAP-to-Gaussian similarity
transform happen before this derivation. Select an upright input image whenever
possible. The resulting `world_up` remains fixed for the lifetime of the MCP
process and is returned by `gs_get_pose`.

Persistent campaigns store this axis. When a campaign is resumed, its stored
`world_up` is restored before navigation so a different startup-image tilt
cannot silently change the campaign's movement and heading frame.

`gs_set_pose` changes the raw pose without redefining `world_up`. This is
intentional: a debug teleport must not silently rotate the navigation frame.
Restart with a different initial pose to establish a different vertical axis.

## Rotation

`gs_rotate(yaw_deg=0, pitch_deg=0)` provides two stable controls:

- Positive yaw turns right around fixed `world_up`.
- Positive pitch looks up around the horizon-level camera-right axis.
- Pitch is clamped to `[-85°, +85°]`.
- Roll is not exposed.

After each rotation, the camera basis is rebuilt from its forward direction and
fixed `world_up`. Repeated or alternating yaw/pitch actions therefore do not
accumulate side tilt. Recorded transition frames follow this same stable path;
they do not use unconstrained quaternion interpolation for rotation actions.

The pitch limit prevents the forward direction from becoming parallel to
`world_up`, where a unique level-right direction would cease to exist.

## Translation

`gs_move(direction, distance)` accepts six directions. Distances are scene
units, not assumed real-world metres.

| Direction | World-space behavior |
| --- | --- |
| `forward` / `backward` | Along the complete camera view direction; pitching changes height while moving. |
| `left` / `right` | Along camera-right projected onto the plane perpendicular to `world_up`. |
| `up` / `down` | Along fixed `world_up`, independent of camera pitch. |

All six directions use collision checking when a collision backend is loaded.

For `forward` and `backward` only, a shallow downward request (at most 15
degrees) that collides within two collision steps may use a clearance assist.
The runtime first checks the request's horizontal component, then one fixed
`0.12` scene-unit straight diagonal-up variant. Every variant checks the complete
capsule path. A successful response reports `original_ray_collided`,
`horizontal_projection_applied`, and `clearance_adjustment`. The runtime never
snaps or settles downward, and the assist does not apply to `up`, `down`, or
`gs_approach_target`.
`gs_approach_target` is different: it translates along the selected pixel's
explicit 3D depth ray and preserves camera orientation. This lets the Agent
approach a visible lower or higher target without injecting a look-at roll.

## Exploration expectations

Stable vertical controls are not a requirement to remain at human-eye height.
Continuous exploration should cover useful low, middle, and elevated regions.
In particular, a visible coherent path, courtyard, floor, or walkable-looking
surface below should be inspected from a safe lower viewpoint rather than
leaving the whole recording at the starting elevation.

## Raw pose and compatibility boundary

The underlying `CameraPose` remains a position plus a normalized C2W quaternion
in `wxyz` order. Raw poses can represent roll, and standalone renderer smoke
tools may still accept explicit yaw/pitch/roll to construct a test pose. That
does not change the MCP navigation contract: `gs_rotate` exposes yaw and pitch
only, while `gs_set_pose` remains an unchecked debug operation.
