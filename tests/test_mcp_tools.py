"""CPU-only tests of MCP business logic, with no MCP SDK or CUDA required."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from gs_env import CameraIntrinsics, GSEnvironment
from gs_env.types import CameraPose
from gs_mcp.tools import EnvironmentTools, _debug_transition_frame_count
from gs_mcp.campaign import ExplorationCampaign
from gs_mcp.debug_writer import DebugObservationWriter
from test_environment import FakeRenderer


class ImmediatelyBlockedCollision:
    def max_free_body_distance(self, start, direction, distance, radius, body_height, world_up, step) -> float:
        return 0.0


class ShallowGroundCollision:
    def __init__(self, minimum_upward_displacement: float = 0.0) -> None:
        self.minimum_upward_displacement = minimum_upward_displacement

    def max_free_body_distance(self, _start, direction, distance, _radius, _body_height, _world_up, _step) -> float:
        vertical_displacement = float(np.asarray(direction)[1] * distance)
        return distance if vertical_displacement >= self.minimum_upward_displacement - 1e-9 else 0.0


class PartiallyBlockedCollision:
    def max_free_body_distance(self, _start, _direction, distance, _radius, _body_height, _world_up, _step) -> float:
        return min(0.1, distance)


class MCPToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        renderer = FakeRenderer()
        environment = GSEnvironment(renderer)
        environment.load_scene("fake.ply")
        self.tools = EnvironmentTools(environment, CameraIntrinsics(20, 20, 1, 1, 2, 2))

    def test_move_rotate_and_observe_form_agent_loop(self) -> None:
        np.testing.assert_allclose(self.tools.get_pose()["world_up"], [0, 1, 0], atol=1e-6)
        self.tools.rotate(yaw_deg=90)
        result = self.tools.move("forward", 2.0)
        np.testing.assert_allclose(result["pose"]["position"], [2, 0, 0], atol=1e-6)
        self.assertEqual(self.tools.observe().rgb.shape, (2, 2, 3))

    def test_set_pose_and_validation(self) -> None:
        self.assertEqual(self.tools.set_pose([1, 2, 3], [1, 0, 0, 0])["pose"]["position"], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.tools.move("diagonal", 1)
        with self.assertRaises(ValueError):
            self.tools.move("forward", -1)

    def test_debug_recording_intervals_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentTools(self.tools.environment, self.tools.intrinsics, debug_distance_interval=0)

    def test_debug_frames_follow_distance_and_rotation_intervals(self) -> None:
        start = CameraPose(np.zeros(3), np.array([1, 0, 0, 0]))
        translated = CameraPose(np.array([0.51, 0, 0]), np.array([1, 0, 0, 0]))
        turned = CameraPose(np.zeros(3), np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0]))
        self.assertEqual(_debug_transition_frame_count(start, translated, 0.25, 15), 3)
        self.assertEqual(_debug_transition_frame_count(start, turned, 0.25, 15), 6)

    def test_recorded_rotation_frames_keep_the_horizon_level(self) -> None:
        renderer = FakeRenderer()
        environment = GSEnvironment(renderer)
        environment.load_scene("fake.ply")
        with tempfile.TemporaryDirectory() as directory:
            writer = DebugObservationWriter(directory)
            tools = EnvironmentTools(
                environment,
                self.tools.intrinsics,
                debug_writer=writer,
                debug_rotation_interval_deg=10,
            )
            tools.rotate(yaw_deg=90, pitch_deg=45)
        self.assertGreater(len(renderer.rendered_poses), 1)
        for camera_to_world in renderer.rendered_poses:
            rotation = camera_to_world[:3, :3]
            self.assertAlmostEqual(float(np.dot(rotation[:, 0], environment.get_world_up())), 0.0, places=5)
            self.assertGreater(float(np.dot(-rotation[:, 1], environment.get_world_up())), 0.0)

    def test_navigation_guard_requires_recovery_after_repeated_no_progress(self) -> None:
        environment = GSEnvironment(FakeRenderer(), collision_backend=ImmediatelyBlockedCollision())
        environment.load_scene("fake.ply")
        tools = EnvironmentTools(environment, self.tools.intrinsics, max_navigation_actions=5, blocked_retry_limit=2)
        self.assertFalse(tools.move("forward", 1.0)["navigation_guard"]["recovery_required"])
        self.assertTrue(tools.move("forward", 1.0)["navigation_guard"]["recovery_required"])
        with self.assertRaisesRegex(RuntimeError, "same blocked action"):
            tools.move("forward", 1.0)
        tools.rotate(yaw_deg=30)
        self.assertFalse(tools.move("forward", 1.0)["navigation_guard"]["recovery_required"])

    def test_shallow_downward_collision_uses_safe_horizontal_projection(self) -> None:
        environment = GSEnvironment(FakeRenderer(), collision_backend=ShallowGroundCollision())
        environment.load_scene("fake.ply")
        environment.rotate_local(pitch_deg=-10)
        tools = EnvironmentTools(environment, self.tools.intrinsics)
        result = tools.move("forward", 1.0)
        self.assertFalse(result["collided"])
        self.assertTrue(result["horizontal_projection_applied"])
        self.assertEqual(result["clearance_adjustment"], 0.0)
        self.assertAlmostEqual(result["pose"]["position"][1], 0.0, places=6)
        self.assertAlmostEqual(result["executed_distance"], np.cos(np.deg2rad(10)), places=6)

    def test_shallow_downward_collision_uses_single_fixed_diagonal_clearance(self) -> None:
        environment = GSEnvironment(FakeRenderer(), collision_backend=ShallowGroundCollision(0.06))
        environment.load_scene("fake.ply")
        environment.rotate_local(pitch_deg=-10)
        tools = EnvironmentTools(environment, self.tools.intrinsics)
        result = tools.move("forward", 1.0)
        self.assertFalse(result["collided"])
        self.assertTrue(result["horizontal_projection_applied"])
        self.assertAlmostEqual(result["clearance_adjustment"], 0.12, places=6)
        self.assertAlmostEqual(result["pose"]["position"][1], 0.12, places=6)

    def test_shallow_adjustment_does_not_apply_to_steep_or_vertical_motion(self) -> None:
        steep_environment = GSEnvironment(FakeRenderer(), collision_backend=ShallowGroundCollision())
        steep_environment.load_scene("fake.ply")
        steep_environment.rotate_local(pitch_deg=-30)
        steep = EnvironmentTools(steep_environment, self.tools.intrinsics).move("forward", 1.0)
        self.assertTrue(steep["collided"])
        self.assertNotIn("horizontal_projection_applied", steep)

        vertical_environment = GSEnvironment(FakeRenderer(), collision_backend=ImmediatelyBlockedCollision())
        vertical_environment.load_scene("fake.ply")
        vertical = EnvironmentTools(vertical_environment, self.tools.intrinsics).move("down", 1.0)
        self.assertTrue(vertical["collided"])
        self.assertNotIn("horizontal_projection_applied", vertical)

    def test_shallow_adjustment_does_not_replace_meaningful_partial_progress_or_approach(self) -> None:
        partial_environment = GSEnvironment(FakeRenderer(), collision_backend=PartiallyBlockedCollision())
        partial_environment.load_scene("fake.ply")
        partial_environment.rotate_local(pitch_deg=-10)
        partial = EnvironmentTools(partial_environment, self.tools.intrinsics).move("forward", 1.0)
        self.assertTrue(partial["collided"])
        self.assertAlmostEqual(partial["executed_distance"], 0.1)
        self.assertNotIn("horizontal_projection_applied", partial)

        approach_environment = GSEnvironment(FakeRenderer(), collision_backend=ImmediatelyBlockedCollision())
        approach_environment.load_scene("fake.ply")
        approach_tools = EnvironmentTools(approach_environment, self.tools.intrinsics)
        approach, _observation = approach_tools.approach_target([1, 1], 0.0)
        self.assertTrue(approach["collided"])
        self.assertNotIn("horizontal_projection_applied", approach)

    def test_zero_navigation_budget_disables_action_cap(self) -> None:
        tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, max_navigation_actions=0)
        for _ in range(45):
            guard = tools.rotate(yaw_deg=1)["navigation_guard"]
        self.assertIsNone(guard["actions_remaining"])

    def test_full_panorama_rejects_repeated_rotation_until_spatial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            campaign.state["checkpoints"][0]["heading_bins"] = list(range(8))
            result = tools.rotate(yaw_deg=45)
            self.assertTrue(result["exploration"]["rotation_repetition"])
            self.assertTrue(result["navigation_guard"]["recovery_required"])
            with self.assertRaisesRegex(RuntimeError, "full panorama"):
                tools.rotate(yaw_deg=-45)
            tools.move("forward", 1.0)
            self.assertFalse(campaign.exploration_status()["rotation_repetition"])

    def test_full_panorama_allows_one_reorientation_per_untried_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            campaign.state["checkpoints"][0]["heading_bins"] = list(range(8))
            campaign.report_candidates(self.tools.environment.get_pose(), [{"direction_world": [1, 0, 0]}], [])
            tools.rotate(yaw_deg=45)
            result = tools.rotate(yaw_deg=45)
            self.assertEqual(result["exploration"]["suggested_action"], "explore_route_frontier")
            self.assertFalse(result["navigation_guard"]["recovery_required"])
            with self.assertRaisesRegex(RuntimeError, "full panorama"):
                tools.rotate(yaw_deg=360)

    def test_small_route_alignment_is_not_reported_as_rotation_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            angle = np.deg2rad(10.0)
            campaign.report_candidates(
                self.tools.environment.get_pose(),
                [{"direction_world": [float(np.sin(angle)), 0.0, -float(np.cos(angle))]}],
                [],
            )
            result = tools.rotate(yaw_deg=10)
            self.assertFalse(result["exploration"]["rotation_repetition"])
            self.assertFalse(result["navigation_guard"]["recovery_required"])

    def test_route_guidance_allows_one_turn_but_rejects_second_same_place_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            campaign.report_candidates(self.tools.environment.get_pose(), [{"direction_world": [1, 0, 0]}], [])
            tools.rotate(yaw_deg=90)
            with self.assertRaisesRegex(RuntimeError, "spatial progress"):
                tools.rotate(yaw_deg=90)

    def test_scan_yaw_records_swept_bins_without_exposing_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            first = tools.rotate(yaw_deg=90)
            self.assertGreaterEqual(len(campaign.state["checkpoints"][0]["heading_bins"]), 2)
            self.assertNotIn("heading_bins_seen", first["exploration"]["current_checkpoint"])
            with self.assertRaisesRegex(RuntimeError, "no new heading coverage"):
                tools.rotate(yaw_deg=-90)

    def test_rotation_does_not_erase_collision_memory_for_same_world_direction(self) -> None:
        environment = GSEnvironment(FakeRenderer(), collision_backend=ImmediatelyBlockedCollision())
        environment.load_scene("fake.ply")
        tools = EnvironmentTools(environment, self.tools.intrinsics, blocked_retry_limit=2)
        self.assertFalse(tools.move("forward", 1.0)["navigation_guard"]["recovery_required"])
        tools.rotate(yaw_deg=360)
        self.assertTrue(tools.move("forward", 1.0)["navigation_guard"]["recovery_required"])

    def test_normal_quality_record_does_not_block_forward_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            tools.record_finding()
            self.assertGreater(tools.move("forward", 1.0)["executed_distance"], 0)

    def test_boundary_candidate_becomes_blocked_only_after_no_progress_collision(self) -> None:
        environment = GSEnvironment(FakeRenderer(), collision_backend=ImmediatelyBlockedCollision())
        environment.load_scene("fake.ply")
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(environment, self.tools.intrinsics, campaign=campaign)
            tools.record_finding(boundary_candidate=True)
            result = tools.move("forward", 1.0)
            self.assertEqual(result["boundary_route_warning"]["status"], "candidate")
            self.assertEqual(result["boundary_route_status"], "blocked")
            with self.assertRaisesRegex(RuntimeError, "previously confirmed blocked"):
                tools.move("forward", 1.0)

    def test_reports_pixel_candidates_and_returns_exploration_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            result = tools.report_exploration_candidates(
                route_frontiers=[{"pixel": [1, 1], "kind": "doorway", "reason": "visible opening"}],
            )
            self.assertEqual(result["route_frontiers"][0]["status"], "untried")
            status = tools.get_exploration_status()
            self.assertEqual(status["suggested_action"], "explore_route_frontier")
            self.assertEqual(status["campaign"]["frames_recorded"], 0)
            self.assertFalse(status["campaign"]["complete"])
            self.assertEqual(status["campaign"]["completed_video_paths"], [])
            with self.assertRaises(ValueError):
                tools.report_exploration_candidates(route_frontiers=[{"pixel": [5, 1]}])

    def test_configure_campaign_sets_exact_run_clip_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            result = tools.configure_campaign(2, objective="coverage")
            self.assertEqual(result["target_video_clips"], 2)
            self.assertEqual(result["video_clips_remaining"], 2)
            self.assertEqual(result["exploration_objective"], "coverage")

    def test_checkpoint_restore_is_restricted_to_boundary_or_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            campaign.state["frames_recorded"] = 1
            with self.assertRaisesRegex(RuntimeError, "81-frame boundary"):
                tools.restore_checkpoint(0)
            campaign.state["moves_since_new_checkpoint"] = 4
            restored = tools.restore_checkpoint(0, "stuck recovery")
            self.assertEqual(restored["checkpoint_id"], 0)
            self.assertEqual(campaign.state["restore_events"][-1]["reason"], "stuck recovery")

    def test_direct_pose_does_not_create_a_restorable_campaign_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ExplorationCampaign(Path(directory), self.tools.environment.get_pose(), camera_radius=0.25)
            tools = EnvironmentTools(self.tools.environment, self.tools.intrinsics, campaign=campaign)
            tools.set_pose([10, 10, 10], [1, 0, 0, 0])
            tools.observe()
            self.assertEqual(len(campaign.state["checkpoints"]), 1)
            with self.assertRaisesRegex(RuntimeError, "restore a persisted checkpoint"):
                tools.report_exploration_candidates(route_frontiers=[{"pixel": [1, 1]}])
