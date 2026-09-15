# P1 camera controller: coordinate and action contract

`CameraController` owns the current camera pose. It does **not** perform
collision checks in P1; collision will wrap translations later without changing
this coordinate contract.

- Pose storage: world position `[x, y, z]` in meters plus a unit C2W
  quaternion in **wxyz** order.
- C2W maps the renderer's OpenCV/gsplat camera frame (`+X` right, `+Y` image
  down, `+Z` forward) to world coordinates.
- The default pose matches the P0 CLI: it starts at world origin, looks along
  world `-Z`, and has physical/image-up along world `+Y`.
- Session `world_up` defaults to initial image-up and can be supplied explicitly
  when the reconstruction has a known vertical axis.
- `translate_local(forward, right, up)` uses view-forward, horizon-level right,
  and the fixed world-up direction inferred from the initial image.
- `rotate_local(yaw_deg, pitch_deg)` uses horizon-stable navigation rotations:
  positive yaw turns right around fixed world-up, positive pitch looks up, and
  pitch is clamped to `[-85, 85]` degrees. Roll is intentionally unavailable.

`get_camera_to_world()` is ready to pass directly to `GSplatRenderer.render`.
`set_camera_to_world()` validates a proper rigid homogeneous transform, while
`set_pose()` accepts the canonical `CameraPose` representation. The supplied
CPU-only tests verify axes and C2W/W2C round-trips, so they can run identically
on Windows and the future Linux server.
