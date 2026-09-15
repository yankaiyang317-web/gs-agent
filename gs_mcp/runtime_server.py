"""Long-lived MCP server with runtime-selectable remote scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

from gs_mcp.runtime import SceneRuntime
from gs_mcp.server import create_server
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the persistent gs-agent Streamable HTTP service")
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--demo-runs-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug-distance-interval", type=float, default=0.25)
    parser.add_argument("--debug-rotation-interval-deg", type=float, default=15.0)
    parser.add_argument("--max-navigation-actions", type=int, default=0)
    parser.add_argument("--blocked-retry-limit", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18913)
    args = parser.parse_args()
    runtime = SceneRuntime(
        args.scenes_root,
        args.runs_root,
        demo_runs_root=args.demo_runs_root,
        device=args.device,
        debug_distance_interval=args.debug_distance_interval,
        debug_rotation_interval_deg=args.debug_rotation_interval_deg,
        max_navigation_actions=args.max_navigation_actions,
        blocked_retry_limit=args.blocked_retry_limit,
    )
    try:
        server = create_server(
            runtime.proxy,
            scene_runtime=runtime,
            fastmcp_options={"host": args.host, "port": args.port},
        )
        server.run(transport="streamable-http")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
