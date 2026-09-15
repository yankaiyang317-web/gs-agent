"""Export numbered MCP debug RGB frames to short MP4 clips using ffmpeg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gs_mcp.video import export_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Export opt-in MCP debug RGB frames as short MP4 clips")
    parser.add_argument("--session", type=Path, required=True, help="One session_... directory containing rgb/")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=9.0)
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--ffmpeg", default=None)
    args = parser.parse_args()
    output_dir = args.out or args.session / "video"
    for output in export_groups(args.session, output_dir, args.fps, args.max_frames, args.ffmpeg):
        print(output)


if __name__ == "__main__":
    main()
