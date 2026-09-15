"""MP4 export for optional local MCP debug sessions."""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClipExport:
    process: subprocess.Popen
    output: Path
    temporary: Path
    finished: threading.Event
    succeeded: bool = False


_exports: dict[Path, ClipExport] = {}
_exports_lock = threading.Lock()


def frame_groups(rgb_dir: Path, max_frames: int) -> list[list[Path]]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    frames = sorted(rgb_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not frames:
        raise ValueError(f"no PNG frames found in {rgb_dir}")
    return [frames[index:index + max_frames] for index in range(0, len(frames), max_frames)]


def export_groups(session_dir: Path, output_dir: Path, fps: float = 9.0, max_frames: int = 81, ffmpeg: str | None = None) -> list[Path]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    executable = ffmpeg or shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg was not found; activate the Conda environment or pass --ffmpeg PATH")
    groups = frame_groups(session_dir / "rgb", max_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for clip_index, frames in enumerate(groups):
        output = output_dir / f"clip_{clip_index:03d}.mp4"
        command = [executable, "-y", "-framerate", str(fps), "-start_number", str(int(frames[0].stem)), "-i", str(session_dir / "rgb" / "%06d.png"), "-frames:v", str(len(frames)), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        outputs.append(output)
    return outputs


def export_completed_clip_in_background(session_dir: Path, clip_index: int, fps: float = 9.0, max_frames: int = 81, ffmpeg: str | None = None) -> ClipExport | None:
    """Start and track an atomic export of one already-complete clip."""
    executable = ffmpeg or shutil.which("ffmpeg")
    if executable is None:
        return None
    output_dir = session_dir / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"clip_{clip_index:03d}.mp4"
    temporary = output_dir / f"clip_{clip_index:03d}.encoding.mp4"
    key = output.resolve()
    with _exports_lock:
        existing = _exports.get(key)
        if existing is not None and not existing.finished.is_set():
            return existing
    start_number = clip_index * max_frames
    command = [executable, "-nostdin", "-y", "-framerate", str(fps), "-start_number", str(start_number), "-i", str(session_dir / "rgb" / "%06d.png"), "-frames:v", str(max_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    export = ClipExport(process, output, temporary, threading.Event())
    with _exports_lock:
        _exports[key] = export

    def finalize() -> None:
        try:
            returncode = process.wait()
            if returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                temporary.replace(output)
                export.succeeded = True
        finally:
            export.finished.set()

    threading.Thread(target=finalize, name=f"clip-export-{clip_index:03d}", daemon=True).start()
    return export


def wait_for_completed_clip(session_dir: Path, clip_index: int, timeout: float = 300.0) -> bool:
    """Wait until a clip is atomically published and non-empty."""
    output = (session_dir / "video" / f"clip_{clip_index:03d}.mp4").resolve()
    if output.is_file() and output.stat().st_size > 0:
        return True
    with _exports_lock:
        export = _exports.get(output)
    if export is None or not export.finished.wait(timeout):
        return False
    return export.succeeded and output.is_file() and output.stat().st_size > 0


def wait_for_all_exports(timeout: float = 300.0) -> bool:
    """Wait for every managed background encoder before shutdown re-export."""
    with _exports_lock:
        exports = list(_exports.values())
    return all(export.finished.wait(timeout) and export.succeeded for export in exports)
