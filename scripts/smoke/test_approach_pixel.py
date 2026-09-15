"""Real-scene smoke test for depth-based, collision-aware pixel approach."""

from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gs_env import GSEnvironment, load_scene_manifest
from gs_env.collision import create_collision_backend
from gs_env.rendering import GSplatRenderer

manifest = load_scene_manifest('scenes/jiudian.json')
collision = create_collision_backend(
    mesh_path=manifest.collision_mesh_path,
    world_to_asset=manifest.collision_world_to_asset,
)
renderer = GSplatRenderer()
env = GSEnvironment(
    renderer,
    initial_pose=manifest.initial_pose,
    collision_backend=collision,
    camera_radius=manifest.camera_radius,
    camera_body_height=manifest.camera_body_height,
    collision_step=manifest.collision_step,
    world_up=manifest.world_up,
)
env.load_scene(manifest.ply_path)
k = manifest.intrinsics
move, observation, target = env.approach_pixel((160, 120), .5, k)
out = Path('outputs/jiudian_approach_smoke'); out.mkdir(parents=True, exist_ok=True)
Image.fromarray((observation.rgb * 255).round().astype(np.uint8)).save(out / 'rgb.png')
print('target=', target.tolist(), 'executed=', move.executed_distance, 'collided=', move.collided)
