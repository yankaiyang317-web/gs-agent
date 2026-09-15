"""CPU-only tests for per-run exploration campaigns and explicit resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gs_env import CameraIntrinsics, CameraPose, Observation
from gs_env.geometry.transforms import rotation_matrix_to_quaternion
from gs_mcp.campaign import ExplorationCampaign


def _observation() -> Observation:
    return Observation(
        rgb=np.zeros((2, 2, 3), dtype=np.float32),
        depth=np.ones((2, 2), dtype=np.float32),
        alpha=np.ones((2, 2), dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        intrinsics=CameraIntrinsics(1, 1, 1, 1, 2, 2),
    )


class CampaignTests(unittest.TestCase):
    def test_new_campaign_uses_coverage_topology_schema(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), pose, camera_radius=0.25)
            self.assertEqual(campaign.state["version"], 4)
            self.assertEqual(campaign.state["exploration_objective"], "coverage")
            self.assertEqual(len(campaign.state["regions"]), 1)
            self.assertEqual(campaign.state["edges"], [])
            self.assertEqual(campaign.state["checkpoints"][0]["role"], "decision")
            self.assertEqual(campaign.video_segment_frames, 81)
            self.assertEqual(campaign.video_fps, 9)

    def test_v2_campaign_migrates_without_losing_timeline(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = ExplorationCampaign(root, pose, camera_radius=0.25)
            state = campaign.state
            state["version"] = 2
            state["frames_recorded"] = 17
            for key in ("edges", "regions", "route_frontiers", "viewpoint_candidates", "recent_checkpoint_path", "moves_since_new_checkpoint", "restore_events", "exploration_objective"):
                state.pop(key, None)
            for checkpoint in state["checkpoints"]:
                checkpoint.pop("region_id", None)
                checkpoint.pop("role", None)
                checkpoint.pop("role_reason", None)
            campaign.state_path.write_text(json.dumps(state), encoding="utf-8")
            migrated = ExplorationCampaign(root, pose, camera_radius=0.25)
            self.assertEqual(migrated.state["version"], 4)
            self.assertEqual(migrated.state["frames_recorded"], 17)
            self.assertEqual(migrated.state["exploration_objective"], "coverage")
            self.assertEqual(len(migrated.state["regions"]), 1)
            self.assertEqual(migrated.state["checkpoints"][0]["role"], "decision")

    def test_records_continuous_frames_and_starts_completed_clip_export(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, patch("gs_mcp.campaign_base.export_completed_clip_in_background") as export, patch("gs_mcp.campaign_base.wait_for_completed_clip", return_value=True):
            root = Path(directory)
            campaign = ExplorationCampaign(root / "campaign", pose, camera_radius=0.25, target_video_clips=1)
            def publish(*_args, **_kwargs):
                video = root / "campaign" / "video"
                video.mkdir(exist_ok=True)
                (video / "clip_000.mp4").write_bytes(b"ready")
            export.side_effect = publish
            for index in range(81):
                source = root / f"source_{index:03d}.png"
                source.touch()
                self.assertTrue(campaign.record_frame(source, _observation()))
            self.assertTrue(campaign.is_complete)
            self.assertEqual(campaign.status()["recorded_video_clips"], 1)
            self.assertEqual(campaign.status()["completed_video_clips"], 1)
            self.assertTrue((root / "campaign" / "rgb" / "000080.png").is_file())
            export.assert_called_once_with(root / "campaign", 0, fps=9.0, max_frames=81)

    def test_custom_segment_boundary_target_and_fps_are_persisted(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, patch("gs_mcp.campaign_base.export_completed_clip_in_background") as export, patch("gs_mcp.campaign_base.wait_for_completed_clip", return_value=True):
            root = Path(directory) / "demo"
            campaign = ExplorationCampaign(root, pose, camera_radius=0.25, target_video_clips=2, video_segment_frames=270, video_fps=12)
            self.assertEqual(campaign.target_frames, 540)
            for index in range(270):
                source = root.parent / f"demo_{index:03d}.png"
                source.touch()
                campaign.record_frame(source, _observation())
            export.assert_called_once_with(root, 0, fps=12.0, max_frames=270)
            self.assertTrue(campaign.can_restore_checkpoint())
            self.assertEqual(campaign.status()["recorded_video_clips"], 1)
            persisted = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["video_segment_frames"], 270)
            self.assertEqual(persisted["video_fps"], 12.0)

    def test_resume_uses_persisted_video_format_and_old_campaign_defaults(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "persisted"
            ExplorationCampaign(root, pose, 0.25, video_segment_frames=270, video_fps=12)
            resumed = ExplorationCampaign(root, pose, 0.25, video_segment_frames=81, video_fps=9)
            self.assertEqual((resumed.video_segment_frames, resumed.video_fps), (270, 12.0))

            state = resumed.state
            state.pop("video_segment_frames")
            state.pop("video_fps")
            resumed.state_path.write_text(json.dumps(state), encoding="utf-8")
            migrated = ExplorationCampaign(root, pose, 0.25, video_segment_frames=270, video_fps=12)
            self.assertEqual((migrated.video_segment_frames, migrated.video_fps), (81, 9.0))

    def test_custom_checkpoint_boundary_rejects_default_boundary(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), pose, 0.25, video_segment_frames=270)
            campaign.state["frames_recorded"] = 81
            self.assertFalse(campaign.can_restore_checkpoint())
            campaign.state["frames_recorded"] = 270
            self.assertTrue(campaign.can_restore_checkpoint())

    def test_81_frame_boundary_preserves_exploration_memory(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, patch("gs_mcp.campaign_base.export_completed_clip_in_background"):
            root = Path(directory)
            campaign = ExplorationCampaign(root / "campaign", pose, camera_radius=0.25)
            campaign.report_candidates(pose, [{"direction_world": [1, 0, 0], "kind": "side_route"}], [])
            for index in range(81):
                source = root / f"boundary_{index:03d}.png"
                source.touch()
                campaign.record_frame(source, _observation())
            self.assertEqual(campaign.state["frames_recorded"], 81)
            self.assertEqual(campaign.exploration_status()["untried_route_frontiers"], 1)
            self.assertTrue(campaign.can_restore_checkpoint())

    def test_two_clip_run_stops_exactly_at_162_frames(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, patch("gs_mcp.campaign_base.export_completed_clip_in_background") as export, patch("gs_mcp.campaign_base.wait_for_completed_clip", return_value=True):
            root = Path(directory)
            campaign = ExplorationCampaign(root / "run", pose, camera_radius=0.25, target_video_clips=2)

            def publish(_session, clip_index, **_kwargs):
                video = root / "run" / "video"
                video.mkdir(exist_ok=True)
                (video / f"clip_{clip_index:03d}.mp4").write_bytes(b"ready")

            export.side_effect = publish
            for index in range(162):
                source = root / f"two_clips_{index:03d}.png"
                source.touch()
                self.assertTrue(campaign.record_frame(source, _observation()))
            extra = root / "extra.png"
            extra.touch()
            self.assertFalse(campaign.record_frame(extra, _observation()))
            self.assertEqual(campaign.state["frames_recorded"], 162)
            self.assertEqual(campaign.status()["completed_video_clips"], 2)
            self.assertEqual(campaign.status()["video_clips_remaining"], 0)
            self.assertTrue(campaign.is_complete)

    def test_quality_region_is_deduplicated_across_viewing_directions(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        turned_pose = CameraPose(np.zeros(3), np.array([0, 0, 1, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), pose, camera_radius=0.25)
            first = campaign.record_finding(pose)
            self.assertTrue(first["recorded"])
            self.assertNotIn("description", first["finding"])
            self.assertFalse(campaign.record_finding(turned_pose)["recorded"])
            self.assertEqual(campaign.status()["quality_regions_recorded"], 1)

    def test_resume_prefers_checkpoint_with_more_unseen_headings(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        other = CameraPose(np.array([2.0, 0.0, 0.0]), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.record_pose(initial)
            campaign.record_pose(other)
            resumed = campaign.choose_start_pose(initial)
            np.testing.assert_allclose(resumed.position, other.position)

    def test_candidates_take_priority_and_exhaustion_scans_unseen_headings(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.record_pose(initial)
            status = campaign.exploration_status()
            self.assertEqual(status["suggested_action"], "scan_for_new_candidates")
            self.assertIn("credible macro route", status["reason"])
            campaign.report_candidates(initial, [{"direction_world": [1, 0, 0]}], [])
            self.assertEqual(campaign.exploration_status()["suggested_action"], "explore_route_frontier")

    def test_status_exposes_only_compact_current_checkpoint_and_region_detail(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.record_pose(initial)
            campaign.report_candidates(initial, [{"direction_world": [1, 0, 0]}], [])
            status = campaign.exploration_status()
            current = status["current_checkpoint"]
            region = status["current_region"]
            self.assertEqual(current["id"], status["current_checkpoint_id"])
            self.assertEqual(current["untried_route_frontiers"], 1)
            self.assertFalse(current["local_route_candidates_exhausted"])
            self.assertEqual(region["id"], status["current_region_id"])
            self.assertEqual(region["checkpoint_count"], 1)
            self.assertNotIn("pose", current)
            self.assertNotIn("heading_bins_seen", current)
            self.assertNotIn("unseen_heading_bins", current)
            self.assertNotIn("scan_complete", current)
            self.assertNotIn("checkpoints_detail", status)

    def test_open_component_macro_route_is_preserved_alongside_narrow_route(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            result = campaign.report_candidates(
                initial,
                [
                    {"direction_world": [1, 0, 0], "kind": "narrow_passage"},
                    {"direction_world": [0, 0, 1], "kind": "open_component_axis"},
                ],
                [],
            )
            self.assertEqual(
                {item["kind"] for item in result["route_frontiers"]},
                {"narrow_passage", "open_component_axis"},
            )
            self.assertEqual(result["exploration"]["untried_route_frontiers"], 2)

    def test_angular_merge_preserves_distinct_evidence_without_exposing_it(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            first = campaign.report_candidates(
                initial,
                [{"direction_world": [1, 0, 0], "kind": "doorway", "reason": "near opening"}],
                [],
            )
            second = campaign.report_candidates(
                initial,
                [{"direction_world": [1, 0.05, 0], "kind": "open_room", "reason": "room visible behind opening"}],
                [],
            )

            self.assertEqual(len(campaign.state["route_frontiers"]), 1)
            candidate = campaign.state["route_frontiers"][0]
            self.assertEqual(candidate["kind"], "doorway")
            self.assertEqual(candidate["reason"], "room visible behind opening")
            self.assertEqual(
                {(entry["kind"], entry["reason"]) for entry in candidate["_evidence"]},
                {("doorway", "near opening"), ("open_room", "room visible behind opening")},
            )
            self.assertNotIn("_evidence", first["route_frontiers"][0])
            self.assertNotIn("_evidence", second["route_frontiers"][0])

    def test_route_merge_is_stricter_than_viewpoint_merge(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        angle = np.deg2rad(26.69)
        second_direction = [float(np.sin(angle)), 0.0, float(np.cos(angle))]
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.report_candidates(initial, [{"direction_world": [0, 0, 1]}], [{"direction_world": [0, 0, 1]}])
            campaign.report_candidates(initial, [{"direction_world": second_direction}], [{"direction_world": second_direction}])
            self.assertEqual(len(campaign.state["route_frontiers"]), 2)
            self.assertEqual(len(campaign.state["viewpoint_candidates"]), 1)

    def test_unproductive_rotation_is_coverage_based_not_frame_based(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, video_segment_frames=270)
            checkpoint = campaign.state["checkpoints"][0]
            checkpoint["heading_bins"] = list(range(8))
            status = campaign.record_rotation_pose(initial)
            self.assertTrue(status["rotation_repetition"])
            self.assertEqual(status["suggested_action"], "spatial_progress_required")
            self.assertEqual(campaign.video_segment_frames, 270)

    def test_transit_checkpoint_backtracks_to_old_frontier_without_filling_headings(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        moved = CameraPose(np.array([0.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.report_candidates(initial, [{"direction_world": [1, 0, 0], "kind": "side_branch"}], [])
            campaign.record_transition(initial, moved, 2.0, 2.0, False)
            status = campaign.exploration_status()
            self.assertEqual(status["current_checkpoint"]["role"], "transit")
            self.assertEqual(status["suggested_action"], "backtrack")
            self.assertEqual(status["suggested_checkpoint_id"], 0)
            self.assertEqual(status["suggested_path"], [1, 0])
            self.assertNotIn("heading_bins_seen", status["current_checkpoint"])

    def test_transit_checkpoint_scans_when_no_saved_route_exists(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        moved = CameraPose(np.array([0.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.record_transition(initial, moved, 2.0, 2.0, False)
            status = campaign.exploration_status()
            self.assertEqual(status["current_checkpoint"]["role"], "transit")
            self.assertEqual(status["suggested_action"], "scan_for_new_candidates")
            self.assertEqual(status["suggested_checkpoint_id"], 1)

    def test_multiple_routes_promote_checkpoint_to_decision(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        moved = CameraPose(np.array([0.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.record_transition(initial, moved, 2.0, 2.0, False)
            campaign.report_candidates(
                moved,
                [
                    {"direction_world": [1, 0, 0], "kind": "side_branch"},
                    {"direction_world": [-1, 0, 0], "kind": "side_branch"},
                ],
                [],
            )
            checkpoint = campaign.state["checkpoints"][1]
            self.assertEqual(checkpoint["role"], "decision")
            self.assertEqual(checkpoint["role_reason"], "multiple_routes")

    def test_single_route_keeps_checkpoint_as_transit(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        moved = CameraPose(np.array([0.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.record_transition(initial, moved, 2.0, 2.0, False)
            campaign.report_candidates(moved, [{"direction_world": [0, 0, 1], "kind": "continuation"}], [])
            self.assertEqual(campaign.state["checkpoints"][1]["role"], "transit")

    def test_collision_promotes_checkpoint_to_decision(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        moved = CameraPose(np.array([0.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.record_transition(initial, moved, 2.0, 2.0, False)
            campaign.record_transition(moved, moved, 1.0, 0.0, True)
            checkpoint = campaign.state["checkpoints"][1]
            self.assertEqual(checkpoint["role"], "decision")
            self.assertEqual(checkpoint["role_reason"], "collision")

    def test_translation_records_coverage_without_observe(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        moved = CameraPose(np.array([0.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.record_transition(initial, moved, 2.0, 2.0, False)
            self.assertEqual(len(campaign.state["checkpoints"]), 2)
            self.assertEqual(len(campaign.state["edges"]), 1)

    def test_same_direction_continuation_adds_neutral_check_without_changing_action(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        first = CameraPose(np.array([2.0, 0.0, 0.0]), initial.quaternion_wxyz)
        second = CameraPose(np.array([4.0, 0.0, 0.0]), initial.quaternion_wxyz)
        third = CameraPose(np.array([6.0, 0.0, 0.0]), initial.quaternion_wxyz)
        turned = CameraPose(np.array([6.0, 0.0, 2.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.report_candidates(initial, [{"direction_world": [0, 0, 1]}], [])
            campaign.record_transition(initial, first, 2.0, 2.0, False)
            campaign.record_transition(first, second, 2.0, 2.0, False)
            self.assertFalse(campaign.exploration_status()["route_continuation_check"]["due"])
            campaign.record_transition(second, third, 2.0, 2.0, False)
            campaign.report_candidates(third, [{"direction_world": [1, 0, 0]}], [])
            status = campaign.exploration_status()
            self.assertTrue(status["route_continuation_check"]["due"])
            self.assertEqual(status["route_continuation_check"]["same_direction_continuations"], 2)
            self.assertEqual(status["suggested_action"], "explore_route_frontier")
            self.assertIn("Long corridors", status["route_continuation_check"]["guidance"])
            campaign.record_transition(third, turned, 2.0, 2.0, False)
            self.assertFalse(campaign.exploration_status()["route_continuation_check"]["due"])
            self.assertGreaterEqual(campaign.exploration_status()["coverage_cells"], 2)

    def test_rotation_records_every_heading_bin_crossed_by_yaw(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            before = set(campaign.state["checkpoints"][0]["heading_bins"])
            campaign.record_rotation_pose(initial, start_pose=initial, yaw_deg=90)
            after = set(campaign.state["checkpoints"][0]["heading_bins"])
            self.assertGreaterEqual(len(after - before), 2)

    def test_meaningful_translation_resets_rotation_streak_but_tiny_motion_does_not(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        tiny = CameraPose(np.array([0.01, 0.0, 0.0]), initial.quaternion_wxyz)
        moved = CameraPose(np.array([2.0, 0.0, 0.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.record_rotation_pose(initial, start_pose=initial, yaw_deg=45)
            campaign.record_transition(initial, tiny, 1.0, 0.01, True)
            self.assertEqual(campaign.state["consecutive_rotations"], 1)
            campaign.record_transition(tiny, moved, 2.0, 1.99, False)
            self.assertEqual(campaign.state["consecutive_rotations"], 0)

    def test_abab_checkpoint_path_is_repetition(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        other = CameraPose(np.array([2.0, 0.0, 0.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.record_transition(initial, other, 2.0, 2.0, False)
            campaign.record_transition(other, initial, 2.0, 2.0, False)
            self.assertFalse(campaign.exploration_status()["recent_repetition"])
            campaign.record_transition(initial, other, 2.0, 2.0, False)
            self.assertTrue(campaign.exploration_status()["recent_repetition"])

    def test_heading_coverage_uses_initial_image_up_not_world_y(self) -> None:
        # Camera image-up is +Z and its initial forward heading is +Y.
        initial_rotation = np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
        initial = CameraPose(np.zeros(3), rotation_matrix_to_quaternion(initial_rotation))
        pitched_forward = np.array([0, 1, 1], dtype=np.float64) / np.sqrt(2)
        pitched_rotation = np.column_stack((
            [1, 0, 0],
            -np.cross([1, 0, 0], pitched_forward),
            pitched_forward,
        ))
        pitched = CameraPose(np.zeros(3), rotation_matrix_to_quaternion(pitched_rotation))
        yawed_rotation = np.column_stack(([0, -1, 0], [0, 0, -1], [1, 0, 0]))
        yawed = CameraPose(np.zeros(3), rotation_matrix_to_quaternion(yawed_rotation))
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25)
            campaign.record_pose(initial)
            campaign.record_pose(pitched)
            self.assertEqual(len(campaign.state["checkpoints"][0]["heading_bins"]), 1)
            campaign.record_pose(yawed)
            self.assertEqual(len(campaign.state["checkpoints"][0]["heading_bins"]), 2)

    def test_resumed_campaign_preserves_its_original_world_up(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ExplorationCampaign(root, pose, camera_radius=0.25, world_up=[0, 0, 1])
            resumed = ExplorationCampaign(root, pose, camera_radius=0.25, world_up=[0, 1, 0])
            np.testing.assert_allclose(resumed.world_up, [0, 0, 1], atol=1e-6)

    def test_clip_target_is_exact_total_for_current_run(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory, patch("gs_mcp.campaign_base.export_completed_clip_in_background") as export:
            root = Path(directory)
            campaign = ExplorationCampaign(root / "campaign", pose, camera_radius=0.25)
            def publish(*_args, **_kwargs):
                video = root / "campaign" / "video"
                video.mkdir(exist_ok=True)
                (video / "clip_000.mp4").write_bytes(b"ready")
            export.side_effect = publish
            for index in range(81):
                source = root / f"source_{index:03d}.png"
                source.touch()
                campaign.record_frame(source, _observation())
            status = campaign.set_clip_target(2)
            self.assertEqual(status["video_clips_remaining"], 1)
            self.assertEqual(status["target_video_clips"], 2)
            self.assertFalse(status["complete"])

    def test_l_shaped_branch_remains_untried_and_recommends_graph_backtrack(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        corner_to_hall = CameraPose(np.array([0.0, 0.0, 1.5]), initial.quaternion_wxyz)
        hall_end = CameraPose(np.array([0.0, 0.0, 3.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.report_candidates(
                initial,
                [
                    {"direction_world": [0, 0, 1], "kind": "hall"},
                    {"direction_world": [1, 0, 0], "kind": "side_branch"},
                ],
                [],
            )
            campaign.record_transition(initial, corner_to_hall, 1.5, 1.5, False)
            campaign.record_transition(corner_to_hall, hall_end, 1.5, 1.5, False)
            self.assertEqual(len(campaign.state["edges"]), 2)
            statuses = {item["kind"]: item["status"] for item in campaign.state["route_frontiers"]}
            self.assertEqual(statuses, {"hall": "explored", "side_branch": "untried"})
            campaign.state["checkpoints"][2]["heading_bins"] = list(range(8))
            status = campaign.exploration_status()
            self.assertEqual(status["suggested_action"], "backtrack")
            self.assertEqual(status["suggested_checkpoint_id"], 0)
            self.assertEqual(status["suggested_path"], [2, 1, 0])

    def test_viewpoint_height_is_discovered_from_accumulated_motion(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        first_step = CameraPose(np.array([0.0, -0.5, 0.0]), initial.quaternion_wxyz)
        lower_view = CameraPose(np.array([0.0, -1.0, 0.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.report_candidates(
                initial,
                [],
                [{"direction_world": [0, -1, 0], "kind": "lower_courtyard", "reason": "coherent path below"}],
            )
            campaign.record_transition(initial, first_step, 0.5, 0.5, False)
            self.assertEqual(campaign.state["viewpoint_candidates"][0]["status"], "active")
            self.assertEqual(campaign.exploration_status()["suggested_action"], "explore_viewpoint_candidate")
            campaign.record_transition(first_step, lower_view, 0.5, 0.5, False)
            self.assertEqual(campaign.state["viewpoint_candidates"][0]["status"], "valuable")
            self.assertEqual(len(campaign.state["regions"][0]["height_samples"]), 2)

    def test_open_space_does_not_generate_vertical_candidates_automatically(self) -> None:
        initial = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        second = CameraPose(np.array([1.5, 0.0, 0.0]), initial.quaternion_wxyz)
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), initial, camera_radius=0.25, world_up=[0, 1, 0])
            campaign.record_transition(initial, second, 1.5, 1.5, False)
            self.assertEqual(campaign.exploration_status()["untried_viewpoint_candidates"], 0)
            self.assertEqual(len(campaign.state["regions"][0]["height_samples"]), 1)
