"""Current coverage-first exploration campaign."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from gs_env.types import CameraPose
from gs_env.geometry.transforms import axis_angle_rotation, quaternion_to_rotation_matrix
from gs_mcp.campaign_base import CampaignBase, HEADING_BINS, VIDEO_FPS, VIDEO_SEGMENT_FRAMES
from gs_mcp.campaign_base import _normalize, _pose_dict, _pose_from_dict


CAMPAIGN_VERSION = 4
EXPLORATION_OBJECTIVES = {"coverage", "reconstruction_quality"}
ROUTE_FRONTIER_MERGE_DEG = 20.0
VIEWPOINT_MERGE_DEG = 30.0
FRONTIER_MATCH_DEG = 40.0
ROUTE_ORIENTATION_DEG = 20.0
DIRECTIONAL_CONTINUATION_DEG = 20.0
DIRECTIONAL_REVIEW_CONTINUATIONS = 2
RECENT_PATH_LIMIT = 12
SCAN_CANDIDATE_REASON = (
    "Scan only until a credible macro route is visible, report it, and make spatial progress; "
    "continue toward a full panorama only when no useful route is visible or this is a major decision point."
)


class ExplorationCampaign(CampaignBase):
    """Add graph, frontier, and region coverage memory to campaign recording."""

    def __init__(
        self,
        root,
        initial_pose: CameraPose,
        camera_radius: float,
        target_video_clips: int | None = None,
        world_up=None,
        video_segment_frames: int = VIDEO_SEGMENT_FRAMES,
        video_fps: float = VIDEO_FPS,
    ) -> None:
        # CampaignBase dispatches set_clip_target dynamically. Defer the target
        # until the coverage fields used by this implementation are present.
        super().__init__(root, initial_pose, camera_radius, None, world_up, video_segment_frames, video_fps)
        self.region_spacing = max(3 * self.checkpoint_spacing, 1.5)
        self.height_merge_distance = max(3 * camera_radius, 0.5)
        self._ensure_current_state(initial_pose)
        if target_video_clips is not None:
            self.set_clip_target(target_video_clips)
        self._save()

    def _ensure_current_state(self, initial_pose: CameraPose) -> None:
        self.state.setdefault("run_id", self.root.name)
        self.state.setdefault("exploration_objective", "coverage")
        self.state.setdefault("edges", [])
        self.state.setdefault("regions", [])
        self.state.setdefault("route_frontiers", [])
        self.state.setdefault("viewpoint_candidates", [])
        self.state.setdefault("recent_checkpoint_path", [])
        self.state.setdefault("moves_since_new_checkpoint", 0)
        self.state.setdefault("unproductive_rotations", 0)
        self.state.setdefault("consecutive_rotations", 0)
        self.state.setdefault("rotation_sweep_degrees", 0.0)
        self.state.setdefault("rotation_yaw_signs", [])
        self.state.setdefault("last_rotation_added_heading", False)
        self.state.setdefault("route_orientation_candidates_used", [])
        self.state.setdefault("restore_events", [])
        checkpoints = self.state.setdefault("checkpoints", [])
        if not checkpoints:
            checkpoints.append(
                {
                    "id": 0,
                    "pose": _pose_dict(initial_pose),
                    "visits": 0,
                    "heading_bins": [],
                    "last_frame": None,
                    "role": "decision",
                    "role_reason": "start",
                }
            )
            self.state["current_checkpoint_id"] = 0
        for checkpoint in checkpoints:
            checkpoint.setdefault("visits", 0)
            checkpoint.setdefault("heading_bins", [])
            checkpoint.setdefault("last_frame", None)
            checkpoint.setdefault("role", "decision" if int(checkpoint["id"]) == 0 else "transit")
            checkpoint.setdefault("role_reason", "start" if int(checkpoint["id"]) == 0 else "translation")
            if "region_id" not in checkpoint:
                checkpoint["region_id"] = self._assign_region(_pose_from_dict(checkpoint["pose"]).position, int(checkpoint["id"]))
            else:
                self._ensure_region_membership(checkpoint)
            # A checkpoint is physical coverage even when it was reached by a
            # translation that was not followed by an explicit observation.
            self._record_coverage_cell(checkpoint, _pose_from_dict(checkpoint["pose"]).position)
        for collection_name in ("route_frontiers", "viewpoint_candidates"):
            for candidate in self.state[collection_name]:
                candidate.setdefault(
                    "_evidence",
                    [
                        {
                            "kind": str(candidate.get("kind", "")),
                            "reason": str(candidate.get("reason", "")),
                            "first_frame": int(candidate.get("created_frame", -1)),
                            "last_frame": int(candidate.get("last_frame", -1)),
                            "sightings": int(candidate.get("sightings", 1)),
                        }
                    ],
                )
        route_counts: dict[int, int] = {}
        for candidate in self.state["route_frontiers"]:
            checkpoint_id = int(candidate["checkpoint_id"])
            route_counts[checkpoint_id] = route_counts.get(checkpoint_id, 0) + 1
        for checkpoint in checkpoints:
            if route_counts.get(int(checkpoint["id"]), 0) >= 2:
                self._promote_checkpoint(checkpoint, "multiple_routes")
        current = int(self.state.get("current_checkpoint_id", checkpoints[0]["id"]))
        if not self.state["recent_checkpoint_path"]:
            self.state["recent_checkpoint_path"] = [current]
        self.state["version"] = CAMPAIGN_VERSION

    def set_clip_target(self, clip_count: int, objective: str | None = None) -> dict[str, Any]:
        if objective is not None:
            self.set_objective(objective, save=False)
        return super().set_clip_target(clip_count)

    def set_additional_clip_target(self, clip_count: int, objective: str | None = None) -> dict[str, Any]:
        """Compatibility alias; clip_count is the exact total for this run."""
        return self.set_clip_target(clip_count, objective)

    def set_objective(self, objective: str, *, save: bool = True) -> None:
        if objective not in EXPLORATION_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(EXPLORATION_OBJECTIVES)}")
        self.state["exploration_objective"] = objective
        if save:
            self._save()

    def choose_start_pose(self, fallback: CameraPose) -> CameraPose:
        checkpoints = self.state["checkpoints"]
        if not checkpoints:
            return fallback
        unfinished = {int(item["checkpoint_id"]) for item in self.state["route_frontiers"] if item["status"] in {"untried", "active"}}
        current_id = self.state.get("current_checkpoint_id")
        ranked = sorted(
            checkpoints,
            key=lambda item: (
                item["id"] in unfinished,
                HEADING_BINS - len(item["heading_bins"]),
                item["id"] == current_id,
                -item["visits"],
            ),
            reverse=True,
        )
        return _pose_from_dict(ranked[0]["pose"])

    def record_pose(self, pose: CameraPose) -> dict[str, Any]:
        """Record a view without interpreting rotation/teleport as an edge."""
        checkpoint = self._ensure_checkpoint(pose)
        self._visit_checkpoint(checkpoint, pose, append_path=False)
        self._save()
        return self.status()

    def record_view_pose(self, pose: CameraPose) -> dict[str, Any]:
        """Update a known checkpoint only; a raw debug teleport cannot create one."""
        checkpoint = self._nearest_checkpoint(pose.position)
        if checkpoint is not None:
            self._visit_checkpoint(checkpoint, pose, append_path=False)
            self._save()
        return self.status()

    def record_rotation_pose(
        self,
        pose: CameraPose,
        *,
        start_pose: CameraPose | None = None,
        yaw_deg: float = 0.0,
    ) -> dict[str, Any]:
        """Record the whole yaw sweep and track rotation independently of coverage."""
        start = pose if start_pose is None else start_pose
        checkpoint = self._nearest_checkpoint(start.position)
        if checkpoint is None:
            return self.exploration_status()
        previous_headings = set(int(value) for value in checkpoint["heading_bins"])
        for heading in self._swept_heading_bins(start, yaw_deg):
            if heading not in checkpoint["heading_bins"]:
                checkpoint["heading_bins"].append(heading)
        self._visit_checkpoint(checkpoint, pose, append_path=False)
        added_heading = any(int(value) not in previous_headings for value in checkpoint["heading_bins"])
        self.state["consecutive_rotations"] = int(self.state.get("consecutive_rotations", 0)) + 1
        self.state["rotation_sweep_degrees"] = float(self.state.get("rotation_sweep_degrees", 0.0)) + abs(float(yaw_deg))
        sign = 0 if abs(float(yaw_deg)) < 1e-9 else (1 if yaw_deg > 0 else -1)
        signs = self.state.setdefault("rotation_yaw_signs", [])
        signs.append(sign)
        del signs[:-3]
        self.state["last_rotation_added_heading"] = added_heading
        if added_heading:
            self.state["unproductive_rotations"] = 0
        else:
            self.state["unproductive_rotations"] = int(self.state.get("unproductive_rotations", 0)) + 1
        self._save()
        return self.exploration_status()

    def rotation_block_reason(self, start_pose: CameraPose, yaw_deg: float) -> str | None:
        """Return why another same-place rotation would be unproductive."""
        count = int(self.state.get("consecutive_rotations", 0))
        if count <= 0:
            return None
        checkpoint = self._nearest_checkpoint(start_pose.position)
        if checkpoint is not None and len({int(value) for value in checkpoint["heading_bins"]}) >= HEADING_BINS:
            return "a full panorama is already recorded at this checkpoint"
        status = self.exploration_status()
        if status["suggested_action"] != "scan_for_new_candidates":
            return "the current exploration action requires spatial progress or backtracking"
        signs = [int(value) for value in self.state.get("rotation_yaw_signs", [])]
        new_sign = 0 if abs(float(yaw_deg)) < 1e-9 else (1 if yaw_deg > 0 else -1)
        if len(signs) >= 2 and signs[-2] == new_sign and signs[-1] == -new_sign and new_sign != 0:
            return "alternating yaw at one checkpoint is repeating the same view"
        if checkpoint is not None:
            swept = set(self._swept_heading_bins(start_pose, yaw_deg))
            unseen = swept.difference(int(value) for value in checkpoint["heading_bins"])
            if not unseen:
                return "this rotation would add no new heading coverage"
        if float(self.state.get("rotation_sweep_degrees", 0.0)) >= 360.0 - 1e-6:
            return "a full panorama is already recorded at this checkpoint"
        return None

    def route_orientation_candidate(self, start_pose: CameraPose, yaw_deg: float) -> int | None:
        """Return a local route that justifies one otherwise-blocked reorientation.

        This is intentionally narrower than a force flag: the proposed yaw must
        end facing an active or untried route recorded at this checkpoint, and
        each route can authorize only one reorientation before spatial progress.
        """
        checkpoint = self._nearest_checkpoint(start_pose.position)
        if checkpoint is None:
            return None
        rotation = quaternion_to_rotation_matrix(start_pose.quaternion_wxyz)
        forward = rotation @ np.array([0.0, 0.0, 1.0])
        turned = axis_angle_rotation(self.world_up, np.deg2rad(-float(yaw_deg))) @ forward
        turned_horizontal = turned - np.dot(turned, self.world_up) * self.world_up
        if np.linalg.norm(turned_horizontal) < 1e-8:
            return None
        used = {int(value) for value in self.state.get("route_orientation_candidates_used", [])}
        matches: list[tuple[float, int]] = []
        for item in self.state.get("route_frontiers", []):
            candidate_id = int(item["id"])
            if (
                int(item["checkpoint_id"]) != int(checkpoint["id"])
                or item["status"] not in {"untried", "active"}
                or candidate_id in used
            ):
                continue
            direction = np.asarray(item["direction_world"], dtype=np.float64)
            horizontal = direction - np.dot(direction, self.world_up) * self.world_up
            if np.linalg.norm(horizontal) < 1e-8:
                continue
            angle = _angle_deg(turned_horizontal, horizontal)
            if angle <= ROUTE_ORIENTATION_DEG:
                matches.append((angle, candidate_id))
        return None if not matches else min(matches)[1]

    def record_route_orientation(self, candidate_id: int) -> None:
        """Consume one route-specific reorientation allowance until translation."""
        used = self.state.setdefault("route_orientation_candidates_used", [])
        if int(candidate_id) not in {int(value) for value in used}:
            used.append(int(candidate_id))
        # This rotation added no visual heading coverage, but it was productive
        # navigation and should not ask the Agent to recover from repetition.
        self.state["unproductive_rotations"] = 0
        self._save()

    def record_transition(
        self,
        start_pose: CameraPose,
        end_pose: CameraPose,
        requested_distance: float,
        executed_distance: float,
        collided: bool,
    ) -> dict[str, Any]:
        """Record one collision-checked physical translation."""
        start = self._ensure_checkpoint(start_pose)
        end = self._ensure_checkpoint(end_pose)
        self._visit_checkpoint(end, end_pose, append_path=True)
        new_checkpoint = int(start["id"]) != int(end["id"])
        if new_checkpoint and executed_distance > 0:
            self._record_edge(int(start["id"]), int(end["id"]), executed_distance)
            self.state["moves_since_new_checkpoint"] = 0
        else:
            self.state["moves_since_new_checkpoint"] = int(self.state.get("moves_since_new_checkpoint", 0)) + 1
        displacement = np.asarray(end_pose.position) - np.asarray(start_pose.position)
        direction = None if np.linalg.norm(displacement) < 1e-8 else _normalize(displacement)
        meaningful = executed_distance >= max(0.05, 0.25 * requested_distance)
        if meaningful:
            self._reset_rotation_tracking()
        self._resolve_route_frontiers(start, end, direction, meaningful, collided)
        self._resolve_viewpoint_candidates(start, end, start_pose, end_pose, direction, meaningful, collided)
        if collided:
            self._promote_checkpoint(end, "collision")
        self._save()
        return self.exploration_status()

    def report_candidates(
        self,
        pose: CameraPose,
        route_frontiers: list[dict[str, Any]],
        viewpoint_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        checkpoint = self._ensure_checkpoint(pose)
        self._visit_checkpoint(checkpoint, pose, append_path=False)
        routes = [self._public_candidate(self._merge_candidate("route_frontiers", checkpoint, item)) for item in route_frontiers]
        views = [self._public_candidate(self._merge_candidate("viewpoint_candidates", checkpoint, item)) for item in viewpoint_candidates]
        route_count = sum(
            int(item["checkpoint_id"]) == int(checkpoint["id"])
            for item in self.state["route_frontiers"]
        )
        if route_count >= 2:
            self._promote_checkpoint(checkpoint, "multiple_routes")
        self._save()
        return {"route_frontiers": routes, "viewpoint_candidates": views, "exploration": self.exploration_status()}

    def exploration_status(self) -> dict[str, Any]:
        current_id = int(self.state.get("current_checkpoint_id", 0))
        checkpoint = self._checkpoint_by_id(current_id)
        region_id = None if checkpoint is None else int(checkpoint["region_id"])
        region = None if region_id is None else self._region_by_id(region_id)
        routes = [item for item in self.state["route_frontiers"] if item["status"] in {"untried", "active"}]
        views = [item for item in self.state["viewpoint_candidates"] if item["status"] in {"untried", "active"}]
        continuation_review = self._directional_continuation_review(current_id, routes)
        local_routes = sorted(
            (item for item in routes if int(item["checkpoint_id"]) == current_id),
            key=lambda item: item["status"] != "active",
        )
        local_views = sorted(
            (item for item in views if int(item["region_id"]) == region_id),
            key=lambda item: item["status"] != "active",
        )
        repeated = self._recent_repetition()
        action = "continue_observing"
        target_checkpoint = None
        target_candidate = None
        path: list[int] = []
        reason = "No reported untried candidate at the current checkpoint."
        scan_checkpoint, scan_path = self._nearest_incomplete_heading_checkpoint(current_id)
        current_heading_incomplete = checkpoint is not None and len(checkpoint.get("heading_bins", [])) < HEADING_BINS
        current_is_decision = checkpoint is not None and checkpoint.get("role") == "decision"
        if local_routes:
            action, target_candidate = "explore_route_frontier", int(local_routes[0]["id"])
            reason = "Continue the active route or try an untried frontier at the current checkpoint."
        elif local_views:
            action, target_candidate = "explore_viewpoint_candidate", int(local_views[0]["id"])
            reason = "Continue the active view change or try an evidence-backed regional viewpoint."
        elif routes and current_is_decision and current_heading_incomplete:
            action = "scan_for_new_candidates"
            target_checkpoint = current_id
            reason = "This decision checkpoint has unseen headings that may contain another branch. " + SCAN_CANDIDATE_REASON
        elif routes:
            target_checkpoint, path = self._nearest_unfinished_checkpoint(current_id, routes)
            action = "backtrack" if target_checkpoint is not None else "continue_observing"
            reason = "Return to the nearest checkpoint with an untried route frontier."
        elif current_heading_incomplete:
            action = "scan_for_new_candidates"
            target_checkpoint = current_id
            reason = SCAN_CANDIDATE_REASON
        else:
            if scan_checkpoint is not None:
                action = "backtrack_for_scan"
                target_checkpoint, path = scan_checkpoint, scan_path
                reason = "All candidates are exhausted; revisit the nearest checkpoint with unseen headings."
            elif int(self.state.get("unproductive_rotations", 0)) > 0:
                action = "spatial_progress_required"
                reason = "This checkpoint already has a full panorama; another rotation added no coverage."
        if repeated and action == "continue_observing":
            action = "scan_for_new_candidates"
            reason = "Recent physical motion is repeating; scan once and report a distinct macro route or backtrack."
        current_checkpoint = {
            "id": current_id,
            "role": None if checkpoint is None else checkpoint.get("role", "transit"),
            "visits": 0 if checkpoint is None else int(checkpoint.get("visits", 0)),
            "untried_route_frontiers": sum(item["status"] == "untried" for item in local_routes),
            "active_route_frontiers": sum(item["status"] == "active" for item in local_routes),
            "local_route_candidates_exhausted": not local_routes,
        }
        current_region = {
            "id": region_id,
            "checkpoint_count": 0 if region is None else len(region.get("checkpoint_ids", [])),
            "coverage_cells": 0 if region is None else len(region.get("coverage_cells", [])),
            "untried_viewpoint_candidates": sum(item["status"] == "untried" for item in local_views),
            "active_viewpoint_candidates": sum(item["status"] == "active" for item in local_views),
            "local_viewpoint_candidates_exhausted": not local_views,
        }
        return {
            "objective": self.state["exploration_objective"],
            "current_checkpoint_id": current_id,
            "current_region_id": region_id,
            "current_checkpoint": current_checkpoint,
            "current_region": current_region,
            "checkpoints": len(self.state["checkpoints"]),
            "edges": len(self.state["edges"]),
            "coverage_regions": len(self.state["regions"]),
            "coverage_cells": sum(len(region.get("coverage_cells", [])) for region in self.state["regions"]),
            "untried_route_frontiers": sum(item["status"] == "untried" for item in routes),
            "active_route_frontiers": sum(item["status"] == "active" for item in routes),
            "untried_viewpoint_candidates": sum(item["status"] == "untried" for item in views),
            "active_viewpoint_candidates": sum(item["status"] == "active" for item in views),
            "recent_repetition": repeated,
            "moves_since_new_checkpoint": int(self.state.get("moves_since_new_checkpoint", 0)),
            "unproductive_rotations": int(self.state.get("unproductive_rotations", 0)),
            "consecutive_rotations": int(self.state.get("consecutive_rotations", 0)),
            "rotation_repetition": int(self.state.get("unproductive_rotations", 0)) > 0,
            "route_continuation_check": continuation_review,
            "suggested_action": action,
            "suggested_checkpoint_id": target_checkpoint,
            "suggested_candidate_id": target_candidate,
            "suggested_path": path,
            "reason": reason,
        }

    def checkpoint_pose(self, checkpoint_id: int) -> CameraPose:
        checkpoint = self._checkpoint_by_id(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"unknown checkpoint_id {checkpoint_id}")
        return _pose_from_dict(checkpoint["pose"])

    def can_restore_checkpoint(self) -> bool:
        frames = int(self.state["frames_recorded"])
        return frames % self.video_segment_frames == 0 or self._recent_repetition()

    def record_checkpoint_restore(self, checkpoint_id: int, reason: str) -> dict[str, Any]:
        checkpoint = self._checkpoint_by_id(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"unknown checkpoint_id {checkpoint_id}")
        self.state["current_checkpoint_id"] = int(checkpoint_id)
        self.state["last_pose"] = checkpoint["pose"]
        self.state["recent_checkpoint_path"] = [int(checkpoint_id)]
        self.state["moves_since_new_checkpoint"] = 0
        self._reset_rotation_tracking()
        self.state["restore_events"].append(
            {"checkpoint_id": int(checkpoint_id), "frame": int(self.state["frames_recorded"]) - 1, "reason": reason.strip() or "recovery"}
        )
        self._save()
        return self.exploration_status()

    def status(self) -> dict[str, Any]:
        result = super().status()
        result.update(
            run_id=self.state.get("run_id", self.root.name),
            exploration_objective=self.state.get("exploration_objective", "coverage"),
            edges=len(self.state.get("edges", [])),
            coverage_regions=len(self.state.get("regions", [])),
            untried_route_frontiers=sum(item["status"] == "untried" for item in self.state.get("route_frontiers", [])),
            untried_viewpoint_candidates=sum(item["status"] == "untried" for item in self.state.get("viewpoint_candidates", [])),
        )
        return result

    def _ensure_checkpoint(self, pose: CameraPose) -> dict[str, Any]:
        checkpoint = self._nearest_checkpoint(pose.position)
        if checkpoint is not None:
            return checkpoint
        checkpoint_id = max((int(item["id"]) for item in self.state["checkpoints"]), default=-1) + 1
        region_id = self._assign_region(pose.position, checkpoint_id)
        checkpoint = {
            "id": checkpoint_id,
            "pose": _pose_dict(pose),
            "visits": 0,
            "heading_bins": [],
            "last_frame": None,
            "region_id": region_id,
            "role": "transit",
            "role_reason": "translation",
        }
        self.state["checkpoints"].append(checkpoint)
        return checkpoint

    @staticmethod
    def _promote_checkpoint(checkpoint: dict[str, Any], reason: str) -> None:
        """Permanently mark a topology-relevant checkpoint as a decision point."""
        if checkpoint.get("role") != "decision":
            checkpoint["role"] = "decision"
            checkpoint["role_reason"] = reason

    def _visit_checkpoint(self, checkpoint: dict[str, Any], pose: CameraPose, *, append_path: bool) -> None:
        heading = self._heading_bin(pose)
        if heading not in checkpoint["heading_bins"]:
            checkpoint["heading_bins"].append(heading)
        checkpoint["visits"] += 1
        checkpoint["last_frame"] = int(self.state["frames_recorded"]) - 1 if self.state["frames_recorded"] else None
        self.state["last_pose"] = _pose_dict(pose)
        self.state["current_checkpoint_id"] = int(checkpoint["id"])
        self._record_coverage_cell(checkpoint, pose.position)
        if append_path:
            path = self.state["recent_checkpoint_path"]
            if not path or int(path[-1]) != int(checkpoint["id"]):
                path.append(int(checkpoint["id"]))
                del path[:-RECENT_PATH_LIMIT]

    def _record_edge(self, first_id: int, second_id: int, distance: float) -> None:
        low, high = sorted((first_id, second_id))
        for edge in self.state["edges"]:
            if int(edge["a"]) == low and int(edge["b"]) == high:
                edge["traversals"] = int(edge.get("traversals", 0)) + 1
                edge["last_frame"] = int(self.state["frames_recorded"]) - 1
                edge["last_distance"] = float(distance)
                return
        self.state["edges"].append(
            {"id": len(self.state["edges"]), "a": low, "b": high, "traversals": 1, "last_distance": float(distance), "last_frame": int(self.state["frames_recorded"]) - 1}
        )

    def _merge_candidate(self, collection_name: str, checkpoint: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        direction = _normalize(np.asarray(item["direction_world"], dtype=np.float64))
        collection = self.state[collection_name]
        anchor_key = "checkpoint_id" if collection_name == "route_frontiers" else "region_id"
        anchor_value = int(checkpoint["id"] if anchor_key == "checkpoint_id" else checkpoint["region_id"])
        merge_degrees = ROUTE_FRONTIER_MERGE_DEG if collection_name == "route_frontiers" else VIEWPOINT_MERGE_DEG
        for existing in collection:
            if int(existing[anchor_key]) == anchor_value and _angle_deg(existing["direction_world"], direction) <= merge_degrees:
                existing["sightings"] = int(existing.get("sightings", 1)) + 1
                existing["last_frame"] = int(self.state["frames_recorded"]) - 1
                self._record_candidate_evidence(existing, item)
                if item.get("reason"):
                    existing["reason"] = str(item["reason"])
                return existing
        candidate = {
            "id": len(collection),
            anchor_key: anchor_value,
            "origin_checkpoint_id": int(checkpoint["id"]),
            "direction_world": direction.tolist(),
            "kind": str(item.get("kind", "spatial_frontier" if collection_name == "route_frontiers" else "viewpoint")),
            "reason": str(item.get("reason", "")),
            "status": "untried",
            "sightings": 1,
            "collision_count": 0,
            "created_frame": int(self.state["frames_recorded"]) - 1,
            "last_frame": int(self.state["frames_recorded"]) - 1,
            "origin_height": float(np.dot(_pose_from_dict(checkpoint["pose"]).position, self.world_up)),
            "_evidence": [],
        }
        self._record_candidate_evidence(candidate, item)
        if collection_name == "route_frontiers":
            candidate["target_checkpoint_id"] = None
        collection.append(candidate)
        return candidate

    def _record_candidate_evidence(self, candidate: dict[str, Any], item: dict[str, Any]) -> None:
        """Keep bounded merge evidence internally without expanding Agent responses."""
        kind = str(item.get("kind", candidate.get("kind", "")))
        reason = str(item.get("reason", ""))
        frame = int(self.state["frames_recorded"]) - 1
        evidence = candidate.setdefault("_evidence", [])
        for entry in evidence:
            if entry.get("kind") == kind and entry.get("reason") == reason:
                entry["last_frame"] = frame
                entry["sightings"] = int(entry.get("sightings", 1)) + 1
                return
        if len(evidence) < 8:
            evidence.append({"kind": kind, "reason": reason, "first_frame": frame, "last_frame": frame, "sightings": 1})

    @staticmethod
    def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        """Return the stable public candidate shape; underscore fields stay server-side."""
        return {key: value for key, value in candidate.items() if not key.startswith("_")}

    def _resolve_route_frontiers(self, start, end, direction, meaningful: bool, collided: bool) -> None:
        if direction is None:
            return
        for item in self.state["route_frontiers"]:
            if int(item["checkpoint_id"]) != int(start["id"]) or item["status"] not in {"untried", "active"}:
                continue
            if _angle_deg(item["direction_world"], direction) > FRONTIER_MATCH_DEG:
                continue
            item["last_frame"] = int(self.state["frames_recorded"]) - 1
            if meaningful:
                item["status"] = "explored" if int(start["id"]) != int(end["id"]) else "active"
                if item["status"] == "explored":
                    item["target_checkpoint_id"] = int(end["id"])
            elif collided:
                item["collision_count"] = int(item.get("collision_count", 0)) + 1
                if item["collision_count"] >= 2:
                    item["status"] = "blocked"

    def _resolve_viewpoint_candidates(self, start, end, start_pose, end_pose, direction, meaningful: bool, collided: bool) -> None:
        if direction is None:
            return
        region_id = int(start["region_id"])
        for item in self.state["viewpoint_candidates"]:
            if int(item["region_id"]) != region_id or item["status"] not in {"untried", "active"}:
                continue
            if _angle_deg(item["direction_world"], direction) > FRONTIER_MATCH_DEG:
                continue
            item["last_frame"] = int(self.state["frames_recorded"]) - 1
            height_delta = abs(float(np.dot(end_pose.position, self.world_up)) - float(item.get("origin_height", np.dot(start_pose.position, self.world_up))))
            if meaningful and (height_delta >= self.height_merge_distance or int(end["region_id"]) != region_id):
                item["status"] = "valuable"
                self._record_height_sample(int(end["region_id"]), end_pose.position)
            elif meaningful:
                item["status"] = "active"
            elif collided:
                item["collision_count"] = int(item.get("collision_count", 0)) + 1
                if item["collision_count"] >= 2:
                    item["status"] = "invalid"

    def _assign_region(self, position: np.ndarray, checkpoint_id: int) -> int:
        pos = np.asarray(position, dtype=np.float64)
        nearest = None
        nearest_distance = float("inf")
        for region in self.state.get("regions", []):
            anchor = np.asarray(region["anchor_position"], dtype=np.float64)
            delta = pos - anchor
            horizontal = delta - np.dot(delta, self.world_up) * self.world_up
            distance = float(np.linalg.norm(horizontal))
            if distance < nearest_distance:
                nearest, nearest_distance = region, distance
        if nearest is not None and nearest_distance <= self.region_spacing:
            if checkpoint_id not in nearest.setdefault("checkpoint_ids", []):
                nearest["checkpoint_ids"].append(checkpoint_id)
            nearest.setdefault("coverage_cells", [])
            nearest.setdefault("height_samples", [])
            return int(nearest["id"])
        region_id = max((int(item["id"]) for item in self.state.get("regions", [])), default=-1) + 1
        self.state.setdefault("regions", []).append(
            {
                "id": region_id,
                "anchor_position": pos.tolist(),
                "checkpoint_ids": [checkpoint_id],
                "coverage_cells": [],
                "height_samples": [{"height": float(np.dot(pos, self.world_up)), "observations": 1}],
            }
        )
        return region_id

    def _ensure_region_membership(self, checkpoint: dict[str, Any]) -> None:
        region = self._region_by_id(int(checkpoint["region_id"]))
        if region is None:
            checkpoint["region_id"] = self._assign_region(_pose_from_dict(checkpoint["pose"]).position, int(checkpoint["id"]))
        elif int(checkpoint["id"]) not in region.setdefault("checkpoint_ids", []):
            region["checkpoint_ids"].append(int(checkpoint["id"]))
        if region is not None:
            region.setdefault("coverage_cells", [])
            region.setdefault("height_samples", [])

    def _record_coverage_cell(self, checkpoint: dict[str, Any], position: np.ndarray) -> None:
        region = self._region_by_id(int(checkpoint["region_id"]))
        if region is None:
            return
        cell = [int(np.floor(float(value) / self.checkpoint_spacing)) for value in np.asarray(position)]
        if cell not in region["coverage_cells"]:
            region["coverage_cells"].append(cell)

    def _record_height_sample(self, region_id: int, position: np.ndarray) -> None:
        region = self._region_by_id(region_id)
        if region is None:
            return
        height = float(np.dot(np.asarray(position), self.world_up))
        for sample in region["height_samples"]:
            if abs(float(sample["height"]) - height) <= self.height_merge_distance:
                count = int(sample.get("observations", 1))
                sample["height"] = (float(sample["height"]) * count + height) / (count + 1)
                sample["observations"] = count + 1
                return
        region["height_samples"].append({"height": height, "observations": 1})

    def _checkpoint_by_id(self, checkpoint_id: int) -> dict[str, Any] | None:
        return next((item for item in self.state["checkpoints"] if int(item["id"]) == int(checkpoint_id)), None)

    def _region_by_id(self, region_id: int) -> dict[str, Any] | None:
        return next((item for item in self.state["regions"] if int(item["id"]) == int(region_id)), None)

    def _recent_repetition(self) -> bool:
        path = [int(value) for value in self.state.get("recent_checkpoint_path", [])]
        alternating = len(path) >= 4 and path[-4] == path[-2] and path[-3] == path[-1] and path[-4] != path[-3]
        return int(self.state.get("moves_since_new_checkpoint", 0)) >= 4 or alternating or (len(path) >= 6 and len(set(path[-6:])) <= 2)

    def _directional_continuation_review(self, current_id: int, routes: list[dict[str, Any]]) -> dict[str, Any]:
        """Describe repeated same-direction progress without changing navigation priority."""
        path: list[int] = []
        for value in self.state.get("recent_checkpoint_path", []):
            checkpoint_id = int(value)
            if not path or path[-1] != checkpoint_id:
                path.append(checkpoint_id)
        if not path or path[-1] != int(current_id):
            return {"due": False, "same_direction_continuations": 0, "pending_alternative_routes": 0}

        directions: list[np.ndarray] = []
        for first_id, second_id in zip(path, path[1:]):
            first = self._checkpoint_by_id(first_id)
            second = self._checkpoint_by_id(second_id)
            if first is None or second is None:
                continue
            displacement = _pose_from_dict(second["pose"]).position - _pose_from_dict(first["pose"]).position
            if np.linalg.norm(displacement) >= 1e-8:
                directions.append(_normalize(displacement))

        continuations = 0
        for previous, current in zip(reversed(directions[:-1]), reversed(directions[1:])):
            if _angle_deg(previous, current) > DIRECTIONAL_CONTINUATION_DEG:
                break
            continuations += 1
        pending_alternatives = sum(int(item["checkpoint_id"]) != int(current_id) for item in routes)
        due = continuations >= DIRECTIONAL_REVIEW_CONTINUATIONS
        result: dict[str, Any] = {
            "due": due,
            "same_direction_continuations": continuations,
            "pending_alternative_routes": pending_alternatives,
        }
        if due:
            result["guidance"] = (
                "Check whether coherent scene structure still extends ahead. Long corridors and low-quality but meaningful "
                "regions remain valid, so continue when the fresh view supports them. Prefer another saved route only when "
                "the view indicates a likely scene-ending boundary, after safely observing that boundary when useful."
            )
        return result

    def _reset_rotation_tracking(self) -> None:
        self.state["unproductive_rotations"] = 0
        self.state["consecutive_rotations"] = 0
        self.state["rotation_sweep_degrees"] = 0.0
        self.state["rotation_yaw_signs"] = []
        self.state["last_rotation_added_heading"] = False
        self.state["route_orientation_candidates_used"] = []

    def _swept_heading_bins(self, start_pose: CameraPose, yaw_deg: float) -> list[int]:
        """Return every 45-degree heading sector crossed by a stable yaw."""
        rotation = quaternion_to_rotation_matrix(start_pose.quaternion_wxyz)
        forward = rotation @ np.array([0.0, 0.0, 1.0])
        sample_count = max(1, int(np.ceil(abs(float(yaw_deg)) / (180.0 / HEADING_BINS))))
        headings: list[int] = []
        for index in range(sample_count + 1):
            fraction = index / sample_count
            turned = axis_angle_rotation(self.world_up, np.deg2rad(-float(yaw_deg) * fraction)) @ forward
            heading = self._heading_bin_from_forward(turned)
            if heading not in headings:
                headings.append(heading)
        return headings

    def _nearest_incomplete_heading_checkpoint(self, current_id: int) -> tuple[int | None, list[int]]:
        targets = {
            int(item["id"])
            for item in self.state["checkpoints"]
            if item.get("role") == "decision" and len(item.get("heading_bins", [])) < HEADING_BINS
        }
        return self._nearest_checkpoint_in_set(current_id, targets)

    def _nearest_unfinished_checkpoint(self, current_id: int, frontiers: list[dict[str, Any]]) -> tuple[int | None, list[int]]:
        targets = {int(item["checkpoint_id"]) for item in frontiers}
        return self._nearest_checkpoint_in_set(current_id, targets)

    def _nearest_checkpoint_in_set(self, current_id: int, targets: set[int]) -> tuple[int | None, list[int]]:
        adjacency: dict[int, set[int]] = {}
        for edge in self.state["edges"]:
            a, b = int(edge["a"]), int(edge["b"])
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        queue: deque[tuple[int, list[int]]] = deque([(current_id, [current_id])])
        seen = {current_id}
        while queue:
            node, path = queue.popleft()
            if node in targets:
                return node, path
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        if not targets:
            return None, []
        current = self._checkpoint_by_id(current_id)
        if current is None:
            return None, []
        origin = _pose_from_dict(current["pose"]).position
        target = min(targets, key=lambda item: np.linalg.norm(_pose_from_dict(self._checkpoint_by_id(item)["pose"]).position - origin))
        return target, []


def _angle_deg(first, second) -> float:
    dot = float(np.dot(_normalize(np.asarray(first)), _normalize(np.asarray(second))))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
