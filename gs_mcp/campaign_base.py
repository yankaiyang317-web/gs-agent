"""Persistent exploration memory and continuous segmented video recording."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gs_env.geometry.transforms import quaternion_to_rotation_matrix
from gs_env.types import CameraPose, Observation
from gs_mcp.video import export_completed_clip_in_background, wait_for_completed_clip


VIDEO_SEGMENT_FRAMES = 81
VIDEO_FPS = 9.0
HEADING_BINS = 8


def _pose_dict(pose: CameraPose) -> dict[str, list[float]]:
    return {"position": pose.position.tolist(), "quaternion_wxyz": pose.quaternion_wxyz.tolist()}


def _pose_from_dict(value: dict[str, Any]) -> CameraPose:
    return CameraPose(np.asarray(value["position"], dtype=np.float64), np.asarray(value["quaternion_wxyz"], dtype=np.float64))


class CampaignBase:
    """Store checkpoints, quality regions, and continuous video frames."""

    def __init__(self, root: str | Path, initial_pose: CameraPose, camera_radius: float, target_video_clips: int | None = None, world_up: np.ndarray | list[float] | None = None, video_segment_frames: int = VIDEO_SEGMENT_FRAMES, video_fps: float = VIDEO_FPS) -> None:
        if target_video_clips is not None and target_video_clips <= 0:
            raise ValueError("target_video_clips must be positive when supplied")
        if video_segment_frames <= 0:
            raise ValueError("video_segment_frames must be positive")
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rgb_dir = self.root / "rgb"
        self.rgb_dir.mkdir(exist_ok=True)
        self.state_path = self.root / "campaign.json"
        self.trajectory_path = self.root / "trajectory.jsonl"
        self.checkpoint_spacing = max(4 * camera_radius, 0.5)
        self.finding_radius = max(4 * camera_radius, 0.5)
        initial_rotation = quaternion_to_rotation_matrix(initial_pose.quaternion_wxyz)
        initial_up = _normalize(-initial_rotation[:, 1] if world_up is None else np.asarray(world_up, dtype=np.float64))
        initial_heading = _horizontal_heading(initial_rotation[:, 2], initial_up, initial_rotation[:, 0])
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "version": 2,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "lifecycle_status": "active",
                "ended_at": None,
                "end_reason": None,
                "world_up": initial_up.tolist(),
                "heading_forward": initial_heading.tolist(),
                "target_video_clips": target_video_clips,
                "video_segment_frames": int(video_segment_frames),
                "video_fps": float(video_fps),
                "frames_recorded": 0,
                "last_pose": _pose_dict(initial_pose),
                "current_checkpoint_id": 0,
                "checkpoints": [{"id": 0, "pose": _pose_dict(initial_pose), "visits": 0, "heading_bins": [], "last_frame": None}],
                "findings": [],
                "route_candidates": [],
            }
        self.state.setdefault("route_candidates", [])
        self.state.setdefault("lifecycle_status", "active")
        self.state.setdefault("ended_at", None)
        self.state.setdefault("end_reason", None)
        # Old campaigns keep their historical 81-frame/9 FPS format. For an
        # existing campaign the persisted format is authoritative, regardless
        # of current process defaults or CLI arguments.
        self.state.setdefault("video_segment_frames", VIDEO_SEGMENT_FRAMES)
        self.state.setdefault("video_fps", VIDEO_FPS)
        # Migrate campaigns created before navigation had an explicit stable-up
        # contract. The manifest/input-photo pose passed at startup is the
        # authoritative fallback for those sessions.
        self.state.setdefault("world_up", initial_up.tolist())
        self.state.setdefault("heading_forward", initial_heading.tolist())
        self.state["version"] = max(int(self.state.get("version", 1)), 2)
        self.world_up = _normalize(np.asarray(self.state["world_up"], dtype=np.float64))
        self.heading_forward = _horizontal_heading(
            np.asarray(self.state["heading_forward"], dtype=np.float64), self.world_up, initial_rotation[:, 0]
        )
        self.heading_right = _normalize(np.cross(self.heading_forward, self.world_up))
        if target_video_clips is not None:
            self.set_clip_target(target_video_clips)
        self._start_missing_completed_exports()
        self._save()

    @property
    def target_frames(self) -> int | None:
        target = self.state.get("target_video_clips")
        return None if target is None else int(target) * self.video_segment_frames

    @property
    def video_segment_frames(self) -> int:
        return int(self.state["video_segment_frames"])

    @property
    def video_fps(self) -> float:
        return float(self.state["video_fps"])

    @property
    def is_complete(self) -> bool:
        target = self.target_frames
        target_clips = self.state.get("target_video_clips")
        return target is not None and int(self.state["frames_recorded"]) >= target and self._ready_clip_count() >= int(target_clips)

    def set_clip_target(self, clip_count: int) -> dict[str, Any]:
        """Set the exact number of complete clips required for this run."""
        if not isinstance(clip_count, int) or clip_count <= 0:
            raise ValueError("video clip count must be a positive integer")
        self.state["target_video_clips"] = clip_count
        self.state["requested_video_clips"] = clip_count
        self.state.pop("requested_new_video_clips", None)
        self._save()
        return self.status()

    def set_additional_clip_target(self, clip_count: int) -> dict[str, Any]:
        """Compatibility alias; clip_count is now the exact per-run total."""
        return self.set_clip_target(clip_count)

    def choose_start_pose(self, fallback: CameraPose) -> CameraPose:
        """Resume at a checkpoint with the broadest remaining heading coverage."""
        checkpoints = self.state["checkpoints"]
        if not checkpoints:
            return fallback
        current_id = self.state.get("current_checkpoint_id")
        ranked = sorted(
            checkpoints,
            key=lambda item: (HEADING_BINS - len(item["heading_bins"]), item["id"] == current_id, -item["visits"]),
            reverse=True,
        )
        return _pose_from_dict(ranked[0]["pose"])

    def record_frame(self, source_rgb: Path, observation: Observation, event: dict[str, Any] | None = None) -> bool:
        """Link/copy a session frame into the continuous campaign timeline."""
        if self.is_complete:
            return False
        index = int(self.state["frames_recorded"])
        destination = self.rgb_dir / f"{index:06d}.png"
        try:
            os.link(source_rgb, destination)
        except OSError:
            shutil.copy2(source_rgb, destination)
        record = {"frame": index, "camera_to_world": observation.camera_to_world.tolist(), "intrinsics": vars(observation.intrinsics), "source_frame": str(source_rgb)}
        if event is not None:
            record["event"] = event
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.state["frames_recorded"] = index + 1
        self._save()
        if (index + 1) % self.video_segment_frames == 0:
            clip_index = index // self.video_segment_frames
            export_completed_clip_in_background(self.root, clip_index, fps=self.video_fps, max_frames=self.video_segment_frames)
            if self.target_frames is not None and index + 1 >= self.target_frames:
                if not wait_for_completed_clip(self.root, clip_index):
                    raise RuntimeError(f"final exploration clip {clip_index:03d} did not finish encoding")
                self.mark_lifecycle("completed", "target_video_clips_reached")
        return True

    def set_run_metadata(self, *, scene: str, objective: str) -> None:
        """Persist the task identity used by runtime lifecycle checks."""
        self.state["scene"] = scene
        self.state["exploration_objective"] = objective
        self.state["lifecycle_status"] = "active"
        self.state["ended_at"] = None
        self.state["end_reason"] = None
        self._save()

    def mark_lifecycle(self, status: str, reason: str | None = None) -> None:
        """Persist a non-destructive Campaign lifecycle transition."""
        if status not in {"active", "completed", "interrupted", "abandoned"}:
            raise ValueError("unsupported campaign lifecycle status")
        if self.state.get("lifecycle_status") == status and self.state.get("end_reason") == reason:
            return
        self.state["lifecycle_status"] = status
        terminal = status in {"completed", "interrupted", "abandoned"}
        self.state["ended_at"] = datetime.now().isoformat(timespec="seconds") if terminal else None
        self.state["end_reason"] = reason if terminal else None
        self._save()

    def _ready_clip_count(self) -> int:
        count = 0
        while True:
            output = self.root / "video" / f"clip_{count:03d}.mp4"
            if not output.is_file() or output.stat().st_size <= 0:
                return count
            count += 1

    def _start_missing_completed_exports(self) -> None:
        recorded = int(self.state["frames_recorded"]) // self.video_segment_frames
        for clip_index in range(recorded):
            output = self.root / "video" / f"clip_{clip_index:03d}.mp4"
            if not output.is_file() or output.stat().st_size <= 0:
                export_completed_clip_in_background(self.root, clip_index, fps=self.video_fps, max_frames=self.video_segment_frames)

    def record_pose(self, pose: CameraPose) -> dict[str, Any]:
        checkpoint = self._nearest_checkpoint(pose.position)
        if checkpoint is None:
            checkpoint = {"id": len(self.state["checkpoints"]), "pose": _pose_dict(pose), "visits": 0, "heading_bins": [], "last_frame": None}
            self.state["checkpoints"].append(checkpoint)
        heading = self._heading_bin(pose)
        if heading not in checkpoint["heading_bins"]:
            checkpoint["heading_bins"].append(heading)
        checkpoint["visits"] += 1
        checkpoint["last_frame"] = int(self.state["frames_recorded"]) - 1 if self.state["frames_recorded"] else None
        self.state["last_pose"] = _pose_dict(pose)
        self.state["current_checkpoint_id"] = checkpoint["id"]
        self._save()
        return self.status()

    def record_finding(self, pose: CameraPose, description: str = "", boundary_candidate: bool = False) -> dict[str, Any]:
        """Record one spatially distinct low-quality region; text is optional."""
        text = description.strip()
        for finding in self.state["findings"]:
            old_pose = _pose_from_dict(finding["pose"])
            same_place = np.linalg.norm(old_pose.position - pose.position) <= self.finding_radius
            if same_place:
                finding["seen_count"] += 1
                finding["last_frame"] = int(self.state["frames_recorded"]) - 1
                self._save()
                result = {"recorded": False, "duplicate_of": finding["id"], "campaign": self.status()}
                if boundary_candidate:
                    result["route_candidate"] = self._record_route_candidate(pose)
                    self._save()
                return result
        finding = {"id": len(self.state["findings"]), "pose": _pose_dict(pose), "frame": int(self.state["frames_recorded"]) - 1, "last_frame": int(self.state["frames_recorded"]) - 1, "seen_count": 1}
        if text:
            finding["description"] = text
        self.state["findings"].append(finding)
        route_candidate = self._record_route_candidate(pose) if boundary_candidate else None
        self._save()
        result = {"recorded": True, "finding": finding, "campaign": self.status()}
        if route_candidate is not None:
            result["route_candidate"] = route_candidate
        return result

    def route_guidance(self, pose: CameraPose) -> dict[str, Any] | None:
        """Return the strongest persisted route assessment near this directed pose."""
        matches = [route for route in self.state["route_candidates"] if route["status"] != "cleared" and self._same_route(route, pose)]
        if not matches:
            return None
        return next((route for route in matches if route["status"] == "blocked"), matches[0])

    def resolve_forward_attempt(self, start_pose: CameraPose, requested_distance: float, executed_distance: float, collided: bool) -> dict[str, Any] | None:
        """Confirm or clear a soft boundary candidate using physical motion feedback."""
        route = self.route_guidance(start_pose)
        if route is None or route["status"] != "candidate":
            return route
        meaningful_progress = executed_distance >= max(0.05, 0.25 * requested_distance)
        route["last_frame"] = int(self.state["frames_recorded"]) - 1
        if collided and not meaningful_progress:
            route["status"] = "blocked"
        elif meaningful_progress:
            route["status"] = "cleared"
        self._save()
        return route

    def status(self) -> dict[str, Any]:
        frames = int(self.state["frames_recorded"])
        target = self.state.get("target_video_clips")
        recorded = frames // self.video_segment_frames
        completed = self._ready_clip_count()
        completed_video_paths = [
            str(self.root / "video" / f"clip_{clip_index:03d}.mp4")
            for clip_index in range(completed)
        ]
        return {
            "campaign_id": self.state.get("run_id", self.root.name),
            "scene": self.state.get("scene"),
            "lifecycle_status": "completed" if self.is_complete else self.state.get("lifecycle_status", "active"),
            "end_reason": "target_video_clips_reached" if self.is_complete else self.state.get("end_reason"),
            "campaign_dir": str(self.root),
            "frames_recorded": frames,
            "recorded_video_clips": recorded,
            "completed_video_clips": completed,
            "completed_video_paths": completed_video_paths,
            "target_video_clips": target,
            "target_frames": self.target_frames,
            "video_segment_frames": self.video_segment_frames,
            "video_fps": self.video_fps,
            "video_clips_remaining": None if target is None else max(int(target) - completed, 0),
            "new_video_clips_remaining": None if target is None else max(int(target) - completed, 0),
            "complete": self.is_complete,
            "findings_recorded": len(self.state["findings"]),
            "quality_regions_recorded": len(self.state["findings"]),
            "boundary_route_candidates": sum(route["status"] == "candidate" for route in self.state["route_candidates"]),
            "blocked_routes": sum(route["status"] == "blocked" for route in self.state["route_candidates"]),
            "checkpoints": len(self.state["checkpoints"]),
            "current_checkpoint_id": self.state.get("current_checkpoint_id"),
        }

    def _nearest_checkpoint(self, position: np.ndarray) -> dict[str, Any] | None:
        nearest = None
        nearest_distance = float("inf")
        for checkpoint in self.state["checkpoints"]:
            distance = float(np.linalg.norm(_pose_from_dict(checkpoint["pose"]).position - position))
            if distance < nearest_distance:
                nearest, nearest_distance = checkpoint, distance
        return nearest if nearest_distance <= self.checkpoint_spacing else None

    def _record_route_candidate(self, pose: CameraPose) -> dict[str, Any]:
        for route in self.state["route_candidates"]:
            if route["status"] != "cleared" and self._same_route(route, pose):
                return route
        route = {"id": len(self.state["route_candidates"]), "pose": _pose_dict(pose), "heading_bin": self._heading_bin(pose), "status": "candidate", "frame": int(self.state["frames_recorded"]) - 1, "last_frame": int(self.state["frames_recorded"]) - 1}
        self.state["route_candidates"].append(route)
        return route

    def _same_route(self, route: dict[str, Any], pose: CameraPose) -> bool:
        origin = _pose_from_dict(route["pose"])
        same_place = np.linalg.norm(origin.position - pose.position) <= self.finding_radius
        heading = self._heading_bin(pose)
        stored = int(route["heading_bin"])
        heading_delta = min((heading - stored) % HEADING_BINS, (stored - heading) % HEADING_BINS)
        return same_place and heading_delta <= 1

    def _heading_bin(self, pose: CameraPose) -> int:
        forward = quaternion_to_rotation_matrix(pose.quaternion_wxyz) @ np.array([0.0, 0.0, 1.0])
        return self._heading_bin_from_forward(forward)

    def _heading_bin_from_forward(self, forward: np.ndarray) -> int:
        heading = _horizontal_heading(forward, self.world_up, self.heading_forward)
        angle = np.arctan2(np.dot(heading, self.heading_right), np.dot(heading, self.heading_forward))
        return int(np.floor(((angle + np.pi) / (2 * np.pi)) * HEADING_BINS)) % HEADING_BINS

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-12:
        raise ValueError("campaign direction must be a finite non-zero 3D vector")
    return value / norm


def _horizontal_heading(forward: np.ndarray, world_up: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    projected = np.asarray(forward, dtype=np.float64) - np.dot(forward, world_up) * world_up
    if np.linalg.norm(projected) < 1e-8:
        projected = np.asarray(fallback, dtype=np.float64) - np.dot(fallback, world_up) * world_up
    return _normalize(projected)
