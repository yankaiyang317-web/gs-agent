"""Validation and serialization for minimal MCP tools, independent of FastMCP."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from gs_env import CameraIntrinsics, CameraPose, GSEnvironment, Observation
from gs_env.geometry.unproject import pixel_to_camera_point
from gs_mcp.debug_writer import DebugObservationWriter
from gs_mcp.campaign import ExplorationCampaign


SHALLOW_DOWNWARD_ADJUSTMENT_DEG = 15.0
CLEARANCE_ADJUSTMENT = 0.12


def pose_to_dict(pose: CameraPose) -> dict[str, list[float]]:
    return {"position": pose.position.tolist(), "quaternion_wxyz": pose.quaternion_wxyz.tolist()}


class EnvironmentTools:
    """Agent-facing operations delegated to GSEnvironment; distances are scene units."""

    def __init__(self, environment: GSEnvironment, intrinsics: CameraIntrinsics, debug_writer: DebugObservationWriter | None = None, debug_distance_interval: float = 0.25, debug_rotation_interval_deg: float = 15.0, max_navigation_actions: int = 0, blocked_retry_limit: int = 2, campaign: ExplorationCampaign | None = None) -> None:
        if debug_distance_interval <= 0 or debug_rotation_interval_deg <= 0 or max_navigation_actions < 0 or blocked_retry_limit <= 0:
            raise ValueError("debug intervals and blocked retry limit must be positive; navigation budget must be non-negative")
        self.environment = environment
        self.intrinsics = intrinsics
        self.debug_writer = debug_writer
        self.debug_distance_interval = debug_distance_interval
        self.debug_rotation_interval_deg = debug_rotation_interval_deg
        self.max_navigation_actions = None if max_navigation_actions == 0 else max_navigation_actions
        self.blocked_retry_limit = blocked_retry_limit
        self.campaign = campaign
        self._navigation_actions = 0
        self._blocked_counts: dict[str, int] = {}
        self._direct_pose_untrusted = False

    def get_pose(self) -> dict[str, Any]:
        return {
            "pose": pose_to_dict(self.environment.get_pose()),
            "camera_to_world": self.environment.get_camera_to_world().tolist(),
            "world_up": self.environment.get_world_up().tolist(),
        }

    def set_pose(self, position: list[float], quaternion_wxyz: list[float]) -> dict[str, Any]:
        pose = self.environment.set_pose(CameraPose(np.asarray(position, dtype=np.float64), np.asarray(quaternion_wxyz, dtype=np.float64)))
        self._save_debug_frame({"action": "set_pose"})  # Direct pose changes are intentional teleports.
        # A raw debug teleport is not a validated exploration checkpoint and
        # must not contaminate topology or later checkpoint restore.
        self._direct_pose_untrusted = True
        return {"pose": pose_to_dict(pose), "collision_checked": False}

    def move(self, direction: str, distance: float) -> dict[str, Any]:
        return self._move(direction, distance, action_name="move")

    def _move(self, direction: str, distance: float, action_name: str) -> dict[str, Any]:
        if direction not in {"forward", "backward", "left", "right", "up", "down"}:
            raise ValueError("direction must be forward, backward, left, right, up, or down")
        if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
            raise ValueError("distance must be a finite number greater than or equal to zero")
        keyword, signed_distance = {
            "forward": ("forward", distance), "backward": ("forward", -distance),
            "right": ("right", distance), "left": ("right", -distance),
            "up": ("up", distance), "down": ("up", -distance),
        }[direction]
        start_pose = self.environment.get_pose()
        requested_vector = self.environment.camera.navigation_displacement(**{keyword: signed_distance})
        signature = self._movement_signature(requested_vector)
        route_warning = self._route_warning_or_block(start_pose, direction)
        self._begin_navigation(signature)
        result = self.environment.move_local(**{keyword: signed_distance})
        adjustment = self._try_shallow_downward_adjustment(direction, start_pose, requested_vector, result)
        if adjustment is not None:
            result, adjustment_metadata = adjustment
        else:
            adjustment_metadata = None
        event = {"action": action_name, "direction": direction, "requested_distance": distance, "executed_distance": result.executed_distance, "collided": result.collided}
        if adjustment_metadata is not None:
            event.update(adjustment_metadata)
        self._save_debug_transition(start_pose, result.pose, event)
        route_result = None if direction != "forward" or self.campaign is None else self.campaign.resolve_forward_attempt(start_pose, distance, result.executed_distance, result.collided)
        guard = self._finish_navigation(signature, result.collided and result.executed_distance < max(0.05, 0.25 * distance))
        exploration = self._record_campaign_transition(start_pose, result.pose, distance, result.executed_distance, result.collided)
        response = {"requested": {"action": action_name, "direction": direction, "distance": distance}, "executed_distance": result.executed_distance, "collided": result.collided, "stop_reason": "obstacle" if result.collided else "max_distance", "collision_checked": self.environment.collision_backend is not None, "pose": pose_to_dict(result.pose), "navigation_guard": guard}
        if adjustment_metadata is not None:
            response.update(adjustment_metadata)
            response["stop_reason"] = "adjusted_path"
        if route_warning is not None:
            response["boundary_route_warning"] = route_warning
        if route_result is not None:
            response["boundary_route_status"] = route_result["status"]
        if exploration is not None:
            response["exploration"] = exploration
        return response

    def _try_shallow_downward_adjustment(self, direction: str, start_pose: CameraPose, requested_vector: np.ndarray, original_result) -> tuple[Any, dict[str, Any]] | None:
        """Replace a near-zero shallow descent with one fully checked straight path.

        This is a small clearance assist for the human-sized capsule, not a
        ground or gravity model. It never descends automatically and does not
        apply to vertical movement or depth-target approaches.
        """
        if direction not in {"forward", "backward"} or self.environment.collision_backend is None:
            return None
        if not original_result.collided or original_result.executed_distance > 2 * self.environment.collision_step + 1e-9:
            return None
        vector = np.asarray(requested_vector, dtype=np.float64)
        distance = float(np.linalg.norm(vector))
        if distance < 1e-12:
            return None
        world_up = self.environment.get_world_up()
        vertical_distance = float(np.dot(vector, world_up))
        downward_sine = -vertical_distance / distance
        if downward_sine <= 0 or downward_sine > math.sin(math.radians(SHALLOW_DOWNWARD_ADJUSTMENT_DEG)):
            return None
        horizontal = vector - vertical_distance * world_up
        if np.linalg.norm(horizontal) < 1e-12:
            return None

        original_pose = original_result.pose
        self.environment.set_pose(start_pose)
        candidates = [(0.0, horizontal)]
        candidates.append((CLEARANCE_ADJUSTMENT, horizontal + CLEARANCE_ADJUSTMENT * world_up))
        for clearance, candidate in candidates:
            adjusted = self.environment.move_world(candidate)
            if not adjusted.collided:
                return adjusted, {
                    "original_ray_collided": True,
                    "original_ray_executed_distance": original_result.executed_distance,
                    "horizontal_projection_applied": True,
                    "clearance_adjustment": clearance,
                }
            self.environment.set_pose(start_pose)
        self.environment.set_pose(original_pose)
        return None

    def rotate(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> dict[str, Any]:
        angles = (yaw_deg, pitch_deg)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in angles):
            raise ValueError("rotation angles must be finite numbers")
        start_pose = self.environment.get_pose()
        route_orientation_candidate_id = None
        if self.campaign is not None:
            reason = self.campaign.rotation_block_reason(start_pose, yaw_deg)
            if reason is not None:
                route_orientation_candidate_id = self.campaign.route_orientation_candidate(start_pose, yaw_deg)
                if route_orientation_candidate_id is None:
                    raise RuntimeError(f"{reason}; make spatial progress or backtrack before rotating again")
        self._begin_navigation("rotate")
        pose = self.environment.rotate_local(yaw_deg=yaw_deg, pitch_deg=pitch_deg)
        self._save_debug_transition(start_pose, pose, {"action": "rotate", "yaw_deg": yaw_deg, "pitch_deg": pitch_deg})
        guard = self._finish_navigation("rotate", False)
        exploration = None if self.campaign is None else self.campaign.record_rotation_pose(
            pose,
            start_pose=start_pose,
            yaw_deg=yaw_deg,
        )
        if (
            self.campaign is not None
            and route_orientation_candidate_id is None
            and exploration is not None
            and exploration.get("rotation_repetition")
        ):
            route_orientation_candidate_id = self.campaign.route_orientation_candidate(start_pose, yaw_deg)
        if self.campaign is not None and route_orientation_candidate_id is not None:
            self.campaign.record_route_orientation(route_orientation_candidate_id)
            exploration = self.campaign.exploration_status()
        if exploration is not None and exploration.get("rotation_repetition"):
            guard["recovery_required"] = True
            guard["hint"] = "The full panorama at this checkpoint is already covered; move, follow a candidate, or backtrack instead of rotating again."
        result = {"requested": {"yaw_deg": yaw_deg, "pitch_deg": pitch_deg}, "pose": pose_to_dict(pose), "navigation_guard": guard}
        if exploration is not None:
            result["exploration"] = exploration
        return result

    def observe(self) -> Observation:
        observation = self.environment.observe(self.intrinsics)
        if self.debug_writer is not None:
            self.debug_writer.save(observation, {"action": "observe"})
        self._record_campaign_pose(self.environment.get_pose())
        return observation

    def approach_target(self, pixel: list[int] | None, stop_distance: float, bbox: list[int] | None = None) -> tuple[dict[str, Any], Observation]:
        start_pose = self.environment.get_pose()
        if (pixel is None) == (bbox is None):
            raise ValueError("supply exactly one of pixel or bbox")
        if pixel is not None:
            if len(pixel) != 2:
                raise ValueError("pixel must contain [u, v]")
            selected = {"pixel": pixel}
        else:
            assert bbox is not None
            if len(bbox) != 4:
                raise ValueError("bbox must contain [x0, y0, x1, y1]")
            selected = {"bbox": bbox}
        signature = f"approach:{selected}"
        self._begin_navigation(signature)
        if pixel is not None:
            move, observation, target = self.environment.approach_pixel((int(pixel[0]), int(pixel[1])), stop_distance, self.intrinsics)
        else:
            assert bbox is not None
            move, observation, target = self.environment.approach_bbox(tuple(int(value) for value in bbox), stop_distance, self.intrinsics)
        # approach_target is translation-only.  Keeping the original orientation
        # prevents target depth rays from introducing an implicit camera roll.
        self._save_debug_transition(start_pose, self.environment.get_pose(), {"action": "approach_target", **selected}, final_observation=observation)
        guard = self._finish_navigation(signature, move.collided and move.executed_distance < max(0.05, 0.25 * move.requested_distance))
        end_pose = self.environment.get_pose()
        exploration = self._record_campaign_transition(start_pose, end_pose, move.requested_distance, move.executed_distance, move.collided)
        result = {**selected, "target_world": target.tolist(), "requested_distance": move.requested_distance, "executed_distance": move.executed_distance, "collided": move.collided, "pose": pose_to_dict(end_pose), "navigation_guard": guard}
        if exploration is not None:
            result["exploration"] = exploration
        return (result, observation)

    def record_finding(self, description: str = "", boundary_candidate: bool = False) -> dict[str, Any]:
        """Record a spatially distinct low-quality region at this safe view."""
        if self.campaign is None:
            raise RuntimeError("quality findings require an exploration campaign")
        return self.campaign.record_finding(self.environment.get_pose(), description, boundary_candidate)

    def configure_campaign(self, video_clips: int, objective: str = "coverage") -> dict[str, Any]:
        """Set the exact number of configured video segments for this run."""
        if self.campaign is None:
            raise RuntimeError("campaign configuration requires an exploration campaign")
        return self.campaign.set_clip_target(video_clips, objective=objective)

    def report_exploration_candidates(
        self,
        route_frontiers: list[dict[str, Any]] | None = None,
        viewpoint_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Attach finite, exploration-valued 3D directions to current memory."""
        if self.campaign is None:
            raise RuntimeError("exploration candidates require an exploration campaign")
        if self._direct_pose_untrusted:
            raise RuntimeError("cannot report exploration candidates after gs_set_pose; restore a persisted checkpoint first")
        routes = self._candidate_directions(route_frontiers or [], "route_frontiers")
        views = self._candidate_directions(viewpoint_candidates or [], "viewpoint_candidates")
        if not routes and not views:
            raise ValueError("report at least one route frontier or viewpoint candidate")
        return self.campaign.report_candidates(self.environment.get_pose(), routes, views)

    def get_exploration_status(self) -> dict[str, Any]:
        if self.campaign is None:
            raise RuntimeError("exploration status requires an exploration campaign")
        result = self.campaign.exploration_status()
        result["campaign"] = self.campaign.status()
        return result

    def restore_checkpoint(self, checkpoint_id: int, reason: str = "") -> dict[str, Any]:
        """Restore only a persisted pose, at a clip boundary or during recovery."""
        if self.campaign is None:
            raise RuntimeError("checkpoint restore requires an exploration campaign")
        if self.campaign.is_complete:
            raise RuntimeError("exploration campaign video target reached; do not restore after completion")
        emergency = any(count >= self.blocked_retry_limit for count in self._blocked_counts.values())
        if not self.campaign.can_restore_checkpoint() and not emergency:
            raise RuntimeError(
                f"checkpoint restore is allowed only at a {self.campaign.video_segment_frames}-frame "
                "boundary or after repetition/collision recovery"
            )
        pose = self.campaign.checkpoint_pose(checkpoint_id)
        restored = self.environment.set_pose(pose)
        self._save_debug_frame({"action": "restore_checkpoint", "checkpoint_id": checkpoint_id, "reason": reason})
        exploration = self.campaign.record_checkpoint_restore(checkpoint_id, reason)
        self._blocked_counts.clear()
        self._direct_pose_untrusted = False
        return {"pose": pose_to_dict(restored), "checkpoint_id": checkpoint_id, "collision_checked": "persisted_safe_pose", "exploration": exploration}

    def _save_debug_frame(self, event: dict) -> None:
        """Render an action result only for opt-in local recording, never for Agent output."""
        if self.debug_writer is not None:
            self.debug_writer.save(self.environment.observe(self.intrinsics), event)

    def _save_debug_transition(self, start: CameraPose, end: CameraPose, event: dict, final_observation: Observation | None = None) -> None:
        """Save real renders at spatial/angular intervals only for local recording."""
        if self.debug_writer is None:
            return
        frame_count = _debug_transition_frame_count(start, end, self.debug_distance_interval, self.debug_rotation_interval_deg)
        if event.get("action") == "rotate":
            commanded_angle = max(abs(float(event.get("yaw_deg", 0.0))), abs(float(event.get("pitch_deg", 0.0))))
            frame_count = max(frame_count, int(np.ceil(commanded_angle / self.debug_rotation_interval_deg)))
        for frame_index in range(1, frame_count + 1):
            fraction = frame_index / frame_count
            if event.get("action") == "rotate":
                # Quaternion SLERP between two level endpoints can introduce a
                # visible intermediate roll. Re-run a fraction of the stable
                # yaw/pitch action from the start pose instead.
                self.environment.set_pose(start)
                pose = self.environment.rotate_local(
                    yaw_deg=float(event.get("yaw_deg", 0.0)) * fraction,
                    pitch_deg=float(event.get("pitch_deg", 0.0)) * fraction,
                )
            else:
                pose = CameraPose(
                    (1 - fraction) * start.position + fraction * end.position,
                    _slerp_wxyz(start.quaternion_wxyz, end.quaternion_wxyz, fraction),
                )
                self.environment.set_pose(pose)
            observation = final_observation if frame_index == frame_count and final_observation is not None else self.environment.observe(self.intrinsics)
            self.debug_writer.save(observation, {**event, "interpolation_fraction": fraction})
        self.environment.set_pose(end)

    def _begin_navigation(self, signature: str) -> None:
        if self.campaign is not None and self.campaign.is_complete:
            raise RuntimeError("exploration campaign video target reached; stop and report the completed campaign")
        if self.max_navigation_actions is not None and self._navigation_actions >= self.max_navigation_actions:
            raise RuntimeError("navigation action budget exhausted; stop and report that the target was not found")
        if self._blocked_counts.get(signature, 0) >= self.blocked_retry_limit:
            raise RuntimeError("same blocked action was retried too often; rotate or move in another direction before retrying")
        self._navigation_actions += 1

    def _finish_navigation(self, signature: str, blocked_without_progress: bool) -> dict[str, Any]:
        if blocked_without_progress:
            self._blocked_counts[signature] = self._blocked_counts.get(signature, 0) + 1
        else:
            self._blocked_counts.pop(signature, None)
        recovery_required = self._blocked_counts.get(signature, 0) >= self.blocked_retry_limit
        guard = {"actions_used": self._navigation_actions, "actions_remaining": None if self.max_navigation_actions is None else self.max_navigation_actions - self._navigation_actions, "recovery_required": recovery_required}
        if self.campaign is not None:
            guard["campaign"] = self.campaign.status()
        if recovery_required:
            guard["hint"] = "This direction made almost no progress twice. Observe, rotate, and explore another direction; do not repeat this action unchanged."
        return guard

    @staticmethod
    def _movement_signature(displacement: np.ndarray) -> str:
        """Key collision memory by approximate world direction, not camera-relative wording."""
        vector = np.asarray(displacement, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            return "move:zero"
        quantized = np.round((vector / norm) * 4.0) / 4.0
        quantized[np.abs(quantized) < 1e-12] = 0.0
        return "move:world:" + ",".join(f"{value:.2f}" for value in quantized)

    def _record_campaign_pose(self, pose: CameraPose) -> None:
        if self.campaign is not None and not self._direct_pose_untrusted:
            self.campaign.record_view_pose(pose)

    def _record_campaign_transition(self, start: CameraPose, end: CameraPose, requested: float, executed: float, collided: bool) -> dict[str, Any] | None:
        if self.campaign is None:
            return None
        if self._direct_pose_untrusted:
            return None
        return self.campaign.record_transition(start, end, requested, executed, collided)

    def _candidate_directions(self, candidates: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
        if not isinstance(candidates, list) or len(candidates) > 12:
            raise ValueError(f"{field_name} must be a list with at most 12 entries")
        rotation = self.environment.get_camera_to_world()[:3, :3]
        converted: list[dict[str, Any]] = []
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise ValueError(f"{field_name}[{index}] must be an object")
            pixel = item.get("pixel")
            if not isinstance(pixel, list) or len(pixel) != 2 or not all(isinstance(value, int) for value in pixel):
                raise ValueError(f"{field_name}[{index}].pixel must be integer [u, v]")
            u, v = pixel
            if not (0 <= u < self.intrinsics.width and 0 <= v < self.intrinsics.height):
                raise ValueError(f"{field_name}[{index}].pixel must lie inside the current image")
            camera_ray = pixel_to_camera_point(u, v, 1.0, self.intrinsics)
            world_ray = rotation @ (camera_ray / np.linalg.norm(camera_ray))
            kind = item.get("kind", "spatial_frontier" if field_name == "route_frontiers" else "viewpoint")
            reason = item.get("reason", "")
            if not isinstance(kind, str) or not isinstance(reason, str):
                raise ValueError(f"{field_name}[{index}] kind and reason must be strings")
            converted.append({"direction_world": world_ray.tolist(), "kind": kind.strip(), "reason": reason.strip()})
        return converted

    def _route_warning_or_block(self, pose: CameraPose, direction: str) -> dict[str, Any] | None:
        if self.campaign is None or direction != "forward":
            return None
        route = self.campaign.route_guidance(pose)
        if route is None:
            return None
        if route["status"] == "blocked":
            raise RuntimeError("this forward route was previously confirmed blocked; choose another direction or approach from another side")
        return {"status": "candidate", "hint": "This forward route was visually marked as a possible reconstruction boundary. It remains allowed, but do not push repeatedly; near-zero collision will mark it blocked."}


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path quaternion interpolation for visually smooth debug frames."""
    first = np.asarray(start, dtype=np.float64)
    second = np.asarray(end, dtype=np.float64)
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(np.clip(dot, -1, 1))
    sine = np.sin(angle)
    return (np.sin((1 - fraction) * angle) / sine) * first + (np.sin(fraction * angle) / sine) * second


def _debug_transition_frame_count(start: CameraPose, end: CameraPose, distance_interval: float, rotation_interval_deg: float) -> int:
    """Choose enough real debug renders to bound both positional and angular gaps."""
    distance = float(np.linalg.norm(end.position - start.position))
    dot = abs(float(np.dot(start.quaternion_wxyz, end.quaternion_wxyz)))
    angle_deg = float(np.degrees(2 * np.arccos(np.clip(dot, -1, 1))))
    return max(1, int(np.ceil(distance / distance_interval)), int(np.ceil(angle_deg / rotation_interval_deg)))
