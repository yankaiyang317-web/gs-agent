"""Stable task boundary between MCP transport and scene execution."""

from __future__ import annotations

from enum import Enum
from typing import Any


class HarnessState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"


_ACTIVE_TASK_METHODS = {
    "observe", "get_pose", "set_pose", "move", "approach_target", "rotate",
    "record_finding", "report_exploration_candidates", "get_exploration_status",
    "restore_checkpoint",
}


class HarnessToolsProxy:
    """Expose task tools while enforcing the Harness lifecycle."""

    def __init__(self, harness: "GSAgentHarness") -> None:
        self._harness = harness

    def __getattr__(self, name: str):
        target = getattr(self._harness.runtime.proxy, name)

        def invoke(*args, **kwargs):
            if name in _ACTIVE_TASK_METHODS:
                self._harness.require_active(name)
            result = target(*args, **kwargs)
            self._harness.refresh_state()
            return result

        return invoke


class GSAgentHarness:
    """Coordinate lifecycle deterministically; Agent reasoning stays outside."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.tools = HarnessToolsProxy(self)

    @property
    def state(self) -> HarnessState:
        current = self.runtime.current
        if current is None:
            return HarnessState.IDLE
        return HarnessState.COMPLETED if current.campaign.is_complete else HarnessState.ACTIVE

    def require_active(self, action: str) -> None:
        if self.state is HarnessState.ACTIVE:
            return
        raise RuntimeError(
            f"INVALID_TASK_STATE: {action} requires an active Campaign; "
            f"current_state={self.state.value}; call gs_configure_campaign or gs_get_runtime_status"
        )

    def refresh_state(self) -> HarnessState:
        current = self.runtime.current
        if current is not None and current.campaign.is_complete:
            current.campaign.mark_lifecycle("completed", "target_video_clips_reached")
        return self.state

    def list_scenes(self) -> dict[str, Any]:
        raw = self.runtime.list_scenes()
        scenes = []
        for item in raw["scenes"]:
            safe = {key: value for key, value in item.items() if key not in {"manifest", "error"}}
            if not safe.get("available", False):
                safe["error_code"] = "INVALID_SCENE_MANIFEST"
            scenes.append(safe)
        return {"scenes": scenes, "loaded_scene": raw["loaded_scene"]}

    def describe_scene(self, scene: str) -> dict[str, Any]:
        return self.runtime.describe_scene(scene)

    def get_runtime_status(self) -> dict[str, Any]:
        state = self.refresh_state()
        current = self.runtime.current
        groups = ["discovery", "task_execution"] if state is HarnessState.ACTIVE else ["discovery", "task_start"]
        campaign = None
        if current is not None:
            campaign = current.campaign.status()
            campaign.pop("campaign_dir", None)
        return {
            "task_state": state.value,
            "scene": None if current is None else current.key,
            "campaign": campaign,
            "allowed_tool_groups": groups,
        }

    def configure_campaign(
        self, scene: str, video_clips: int, objective: str,
        profile: str = "development", width: int | None = None,
        height: int | None = None, video_segment_frames: int | None = None,
        video_fps: float | None = None,
    ) -> dict[str, Any]:
        """Start a task or idempotently acknowledge the identical active task."""
        if self.refresh_state() is HarnessState.ACTIVE:
            current = self.runtime.current
            assert current is not None
            settings = self.runtime.campaign_settings(profile, width, height, video_segment_frames, video_fps)
            campaign = current.campaign
            same_task = (
                current.key == scene.strip()
                and current.profile == settings["profile"]
                and current.render_width == settings["width"]
                and current.render_height == settings["height"]
                and campaign.video_segment_frames == settings["video_segment_frames"]
                and campaign.video_fps == settings["video_fps"]
                and campaign.state.get("target_video_clips") == video_clips
                and campaign.state.get("exploration_objective", "coverage") == objective
            )
            if same_task:
                return {"already_active": True, "task_state": "active", "scene": self.runtime.scene_status(), "campaign": campaign.status()}
            raise RuntimeError(
                "ACTIVE_CAMPAIGN_CONFLICT: an incomplete Campaign is already active; "
                "continue it with task-execution tools or ask the server administrator to end it"
            )
        result = self.runtime.configure_campaign(
            scene, video_clips, objective, profile=profile, width=width, height=height,
            video_segment_frames=video_segment_frames, video_fps=video_fps,
        )
        current = self.runtime.current
        assert current is not None
        current.campaign.set_run_metadata(scene=current.key, objective=objective)
        result["campaign"] = current.campaign.status()
        return {"task_state": self.state.value, **result}

    def close(self) -> None:
        current = self.runtime.current
        if current is not None and not current.campaign.is_complete:
            current.campaign.mark_lifecycle("interrupted", "service_shutdown")
        self.runtime.close()
