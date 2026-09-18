# Scene manifests

A scene manifest is the maintained boundary between generic runtime code and
scene-specific data. Asset paths are resolved relative to the JSON file.

Required fields:

- `name`
- `ply`
- `initial_pose.position`
- `initial_pose.quaternion_wxyz` (C2W, wxyz, in PLY world coordinates)
- `camera.width`, `height`, `fx`, `fy`, `cx`, and `cy`

Collision fields:

- `collision_mesh` points to a static GLB triangle mesh. On Linux, Coal builds
  one BVH at scene load and checks one analytic capsule. It is required for
  maintained project scenes.
- `collision.camera_radius` is expressed in scene units.
- `collision.camera_body_height` is the vertical capsule length extending down
  from the camera along `-world_up`. It defaults to `0`, preserving the legacy
  spherical camera. A positive value keeps the camera near a human eye height
  while retaining fully free six-direction flight.
- `collision.step` controls collision sampling, not Agent movement distance.
- `collision.world_to_asset` is an explicit rigid 4x4 transform from renderer
  world coordinates to collision-mesh coordinates. Current splat-transform
  v1.1 meshes declare their Z-180 conversion here instead of hiding it in the
  mesh backend.

`collision.step=0.02` samples one true capsule along the path. Mesh loading is
strict: missing Coal or an invalid GLB is a startup error, never a fallback.
Maintained assets are generated at 0.06 resolution. First run
`--filter-cluster` from the scene's verified initial position, then generate a
`--collision-mesh smooth`; this prevents remote outlier Gaussians from
expanding the working BVH. The reproducible wrapper is
`scripts/generate_collision_mesh.sh`.

Example (the CLI installation path is deployment-specific):

```bash
export SPLAT_TRANSFORM_NODE=/path/to/node
export SPLAT_TRANSFORM_CLI=/path/to/splat-transform/bin/cli.mjs
bash scripts/generate_collision_mesh.sh \
  test_data/my_scene/scene.ply \
  test_data/my_scene/collision_candidates/scene_v001 \
  0,0,0
```

The omitted final argument defaults to `0.06`. Promote the generated GLB to a
versioned formal asset only after visual review and a Coal cold-load/capsule
smoke check with `scripts/tools/validate_collision_mesh.py`.

Optional navigation field:

- `world_up` is a finite non-zero `[x, y, z]` vector in PLY world coordinates.
  It overrides the initial image-up estimate and is normalized by the runtime.

The initial pose should come from a manually verified reconstruction camera.
If its pose is still in COLMAP or viewer coordinates, convert it into final PLY
world coordinates before storing it. Position and orientation must be converted
together.

At environment creation, the runtime derives the session vertical axis from
the initial pose using the OpenCV camera convention:

```python
world_up = -camera_to_world[:3, 1]
```

Choose an initial/input image that is visually upright. A small camera tilt is
retained as a small fixed tilt; it does not accumulate during navigation. A
sideways or upside-down image should be corrected before use. `gs_set_pose`
does not redefine `world_up`; supply a new initial pose at process startup when
the navigation frame itself must change.

An upright but strongly upward- or downward-looking image does not by itself
identify physical gravity. Use manifest `world_up` or CLI `--world-up X Y Z`
when the reconstruction's vertical direction is known independently.

Public documentation intentionally does not enumerate private validation scenes or assets. Keep deployment manifests beside authorized data, or publish redistributable sample assets in a separate dataset repository and link them from this project.

The Windows launcher selects a scene from its first argument or the
`GS_SCENE_MANIFEST` environment variable. Each runtime scene load creates an isolated
run under `outputs/exploration_runs/<manifest-file-name>/run_<timestamp>/`.

Validate a manifest without loading the GPU renderer:

```bash
python -c "from gs_env import load_scene_manifest; print(load_scene_manifest('scenes/my_scene.json'))"
```
