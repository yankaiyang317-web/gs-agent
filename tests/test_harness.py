"""Lifecycle tests for the public Harness boundary."""

import unittest
from types import SimpleNamespace

from gs_mcp.harness import GSAgentHarness, HarnessState


class FakeCampaign:
    def __init__(self, complete=False):
        self.is_complete = complete
        self.video_segment_frames = 81
        self.video_fps = 9.0
        self.state = {"target_video_clips": 2, "exploration_objective": "coverage"}

    def status(self):
        return {"complete": self.is_complete, "target_video_clips": 2}

    def mark_lifecycle(self, status, reason=None):
        self.state.update(lifecycle_status=status, end_reason=reason)

    def set_run_metadata(self, **metadata):
        self.state.update(metadata)


class FakeProxy:
    def move(self, direction, distance):
        return {"direction": direction, "distance": distance}


class FakeRuntime:
    def __init__(self):
        self.current = None
        self.proxy = FakeProxy()

    def list_scenes(self):
        return {"scenes": [{"scene": "demo", "manifest": "/private/demo.json", "available": True}], "loaded_scene": None}

    def describe_scene(self, scene):
        return {"scene": scene, "available": True}

    def scene_status(self):
        return {"scene": self.current.key, "loaded": True}

    def campaign_settings(self, profile, width, height, frames, fps):
        return {"profile": profile, "width": width or 960, "height": height or 720, "video_segment_frames": frames or 81, "video_fps": fps or 9.0}

    def configure_campaign(self, scene, video_clips, objective, **kwargs):
        campaign = FakeCampaign()
        campaign.state.update(target_video_clips=video_clips, exploration_objective=objective)
        self.current = SimpleNamespace(key=scene, profile=kwargs.get("profile", "development"), render_width=960, render_height=720, campaign=campaign)
        return {"scene": {"scene": scene}, "campaign": campaign.status()}

    def close(self):
        pass


class HarnessTests(unittest.TestCase):
    def test_idle_rejects_navigation_and_hides_paths(self):
        harness = GSAgentHarness(FakeRuntime())
        self.assertEqual(harness.state, HarnessState.IDLE)
        self.assertNotIn("manifest", harness.list_scenes()["scenes"][0])
        with self.assertRaisesRegex(RuntimeError, "INVALID_TASK_STATE"):
            harness.tools.move("forward", 1.0)

    def test_start_idempotency_conflict_and_completion(self):
        runtime = FakeRuntime()
        harness = GSAgentHarness(runtime)
        self.assertEqual(harness.configure_campaign("demo", 2, "coverage")["task_state"], "active")
        self.assertEqual(harness.tools.move("forward", 1.0)["distance"], 1.0)
        self.assertTrue(harness.configure_campaign("demo", 2, "coverage")["already_active"])
        with self.assertRaisesRegex(RuntimeError, "ACTIVE_CAMPAIGN_CONFLICT"):
            harness.configure_campaign("other", 2, "coverage")
        runtime.current.campaign.is_complete = True
        self.assertEqual(harness.get_runtime_status()["task_state"], "completed")
        with self.assertRaisesRegex(RuntimeError, "INVALID_TASK_STATE"):
            harness.tools.move("forward", 1.0)


if __name__ == "__main__":
    unittest.main()
