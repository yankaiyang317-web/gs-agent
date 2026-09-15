# P0 renderer conventions and PLY mapping

The renderer uses meters and a right-handed world. `CameraIntrinsics` is the
standard pinhole matrix `K=[[fx,0,cx],[0,fy,cy],[0,0,1]]`, in pixels.

`camera_to_world` (C2W) is a 4x4 homogeneous transform whose rotation maps
the **gsplat/OpenCV camera coordinates** (`+X` right, `+Y` image-down, `+Z`
forward) into world coordinates. The renderer computes `world_to_camera =
inverse(camera_to_world)` and passes that W2C matrix to `gsplat.rasterization`.
At the CLI's zero yaw/pitch/roll, image-up is world `+Y` and the camera looks
towards world `-Z`. Positive CLI yaw turns right, positive pitch turns up, and
positive roll is a right-handed rotation around the camera forward axis.

Canonical 3DGS PLY mapping:

| PLY fields | Renderer tensor / decoding |
| --- | --- |
| `x`, `y`, `z` | Gaussian means in world coordinates |
| `f_dc_0..2` | degree-0 RGB spherical-harmonic coefficients |
| `f_rest_0..` | remaining RGB SH coefficients, rearranged from original-3DGS channel-major order to `[coefficient, RGB]` |
| `opacity` | logit decoded as `sigmoid(opacity)` |
| `scale_0..2` | log-scales decoded as `exp(scale)` |
| `rot_0..3` | normalized Gaussian orientation quaternion in **wxyz** order |

The `RGB+D` gsplat mode is used. `depth` is the accumulated camera-space
z-depth `sum(w_i * z_i)`, not alpha-normalized expected depth. Pixels with
alpha below 0.7 therefore needs to be treated as invalid by downstream
geometry and is shown as black in the MCP depth preview.
`alpha` is gsplat's accumulated opacity, namely the front-to-back composited
coverage `1 - product(1 - alpha_i)` for the contributing Gaussians.

Only canonical original-3DGS PLY is accepted in P0. The loader reports fields
and rejects noncanonical formats clearly rather than silently guessing a field
conversion.
