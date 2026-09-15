"""Run the P1 pose-action sequence without rendering or requiring CUDA."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_env.camera import CameraController


def main() -> None:
    camera = CameraController()
    actions = [
        ("initial", lambda: camera.get_pose()),
        ("forward_1m", lambda: camera.translate_local(forward=1.0)),
        ("right_1m", lambda: camera.translate_local(right=1.0)),
        ("up_1m", lambda: camera.translate_local(up=1.0)),
        ("yaw_right_90deg", lambda: camera.rotate_local(yaw_deg=90.0)),
        ("forward_1m_after_yaw", lambda: camera.translate_local(forward=1.0)),
        ("pitch_up_30deg", lambda: camera.rotate_local(pitch_deg=30.0)),
    ]
    records = []
    for label, action in actions:
        pose = action()
        records.append({"action": label, "position": pose.position.tolist(), "quaternion_wxyz": pose.quaternion_wxyz.tolist(), "camera_to_world": camera.get_camera_to_world().tolist()})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
