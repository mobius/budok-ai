"""FFmpeg-based replay video capture for headless environments.

Supports two modes:
- VM mode (default): starts ffmpeg inside an OrbStack VM and pulls the video
  back to the local run directory.
- Local mode: runs ffmpeg directly on the local machine (set vm_machine to
  "local" or pass --replay-vm local).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReplayCaptureConfig:
    """Configuration for replay video capture."""

    enabled: bool = False
    vm_machine: str = "ubuntu"
    display: str = ":99"
    resolution: str = "1280x720"
    framerate: int = 30
    video_codec: str = "libx264"
    preset: str = "fast"

    @property
    def local_mode(self) -> bool:
        """True when recording should happen on the local host, not inside a VM."""
        return self.vm_machine.strip().lower() == "local"


class ReplayCaptureSession:
    """Manages a single ffmpeg recording session."""

    def __init__(
        self,
        *,
        config: ReplayCaptureConfig,
        match_id: str,
        run_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._match_id = match_id
        self._run_dir = run_dir
        self._logger = logger or logging.getLogger("yomi_daemon.replay_capture")
        self._process: asyncio.subprocess.Process | None = None
        self._vm_video_path = f"/tmp/yomi_replay_{match_id}.mp4"
        self._local_video_path = run_dir / f"_replay_{match_id}.mp4"

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def was_started(self) -> bool:
        """True if recording was started at any point (even if ffmpeg already exited)."""
        return self._process is not None

    async def start_recording(
        self, display: str | None = None, max_duration_seconds: int = 120
    ) -> bool:
        """Start ffmpeg x11grab recording on the configured display."""

        if self._process is not None:
            self._logger.warning("Recording already in progress for match %s", self._match_id)
            return False

        cfg = self._config
        resolved_display = display or cfg.display

        ffmpeg_log = f"/tmp/yomi_ffmpeg_{self._match_id}.log"
        output_path = str(self._local_video_path) if cfg.local_mode else self._vm_video_path

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "x11grab",
            "-video_size",
            cfg.resolution,
            "-framerate",
            str(cfg.framerate),
            "-i",
            resolved_display,
            "-t",
            str(max_duration_seconds),
            "-c:v",
            cfg.video_codec,
            "-preset",
            cfg.preset,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        if cfg.local_mode:
            # Ensure the run directory exists before starting ffmpeg
            self._run_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                self._logger.info(
                    "Started local replay recording for match %s on display %s (pid %s)",
                    self._match_id,
                    resolved_display,
                    self._process.pid,
                )
                return True
            except (OSError, FileNotFoundError) as exc:
                self._logger.error("Failed to start local ffmpeg recording: %s", exc)
                self._process = None
                return False

        if not await self._check_orb_available():
            self._logger.warning("orb command not available — skipping replay recording")
            return False

        orb_ffmpeg_cmd = (
            f"ffmpeg -y -f x11grab "
            f"-video_size {cfg.resolution} "
            f"-framerate {cfg.framerate} "
            f"-i {resolved_display} "
            f"-t {max_duration_seconds} "
            f"-c:v {cfg.video_codec} -preset {cfg.preset} -pix_fmt yuv420p "
            f"{self._vm_video_path} "
            f"</dev/null 2>{ffmpeg_log}"
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                "orb",
                "run",
                "-m",
                cfg.vm_machine,
                "bash",
                "-c",
                orb_ffmpeg_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._logger.info(
                "Started VM replay recording for match %s on display %s (pid %s)",
                self._match_id,
                resolved_display,
                self._process.pid,
            )
            return True
        except (OSError, FileNotFoundError) as exc:
            self._logger.error("Failed to start ffmpeg recording: %s", exc)
            self._process = None
            return False

    async def stop_recording(self) -> Path | None:
        """Stop the ffmpeg recording and make the video available in the run directory."""

        if self._process is None:
            self._logger.warning("No recording in progress for match %s", self._match_id)
            return None

        # ffmpeg was started with -t (fixed duration) so it will exit cleanly
        # on its own. If it already exited, skip waiting.
        if self._process.returncode is None:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=180.0)
                self._logger.info("ffmpeg exited cleanly for match %s", self._match_id)
            except TimeoutError:
                self._logger.warning("ffmpeg did not exit within timeout, killing")
                try:
                    self._process.kill()
                except (ProcessLookupError, OSError):
                    pass
                await self._process.wait()
        else:
            self._logger.info(
                "ffmpeg already exited (code %s) for match %s",
                self._process.returncode,
                self._match_id,
            )

        self._process = None

        # Brief pause for filesystem sync
        await asyncio.sleep(1)

        local_video_path = self._run_dir / "replay.mp4"
        if self._config.local_mode:
            if self._local_video_path.exists():
                try:
                    shutil.move(str(self._local_video_path), str(local_video_path))
                    self._logger.info("Moved local replay video to %s", local_video_path)
                    return local_video_path
                except OSError as exc:
                    self._logger.warning("Failed to move local replay video: %s", exc)
            else:
                self._logger.warning("Local replay video not found at %s", self._local_video_path)
            return None

        # Pull video from VM to local run directory
        return await self._pull_video(local_video_path)

    async def pull_replay_file(self, vm_replay_path: str) -> Path | None:
        """Make the .replay file available in the local run directory."""

        if not vm_replay_path:
            return None

        local_path = self._run_dir / "match.replay"
        try:
            if self._config.local_mode:
                # The mod already globalizes user:// paths to absolute local paths.
                source = Path(vm_replay_path)
                if not source.exists():
                    self._logger.warning("Local replay file does not exist: %s", vm_replay_path)
                    return None
                self._run_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(source), str(local_path))
                self._logger.info("Copied local replay file to %s", local_path)
                return local_path

            proc = await asyncio.create_subprocess_exec(
                "orb",
                "pull",
                "-m",
                self._config.vm_machine,
                vm_replay_path,
                str(local_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            if proc.returncode == 0:
                self._logger.info("Pulled replay file to %s", local_path)
                return local_path
            self._logger.warning(
                "Failed to pull replay file (exit %d): %s",
                proc.returncode,
                stderr.decode(errors="replace").strip(),
            )
        except (OSError, TimeoutError) as exc:
            self._logger.warning("Failed to pull replay file: %s", exc)

        return None

    async def _pull_video(self, local_path: Path) -> Path | None:
        """Pull the video file from VM to local filesystem."""

        try:
            proc = await asyncio.create_subprocess_exec(
                "orb",
                "pull",
                "-m",
                self._config.vm_machine,
                self._vm_video_path,
                str(local_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            if proc.returncode == 0:
                self._logger.info("Pulled replay video to %s", local_path)
                return local_path
            self._logger.warning(
                "Failed to pull video (exit %d): %s",
                proc.returncode,
                stderr.decode(errors="replace").strip(),
            )
        except (OSError, TimeoutError) as exc:
            self._logger.warning("Failed to pull video file: %s", exc)

        return None

    async def _vm_exec(self, command: str) -> int:
        """Run a command in the VM and return exit code."""

        try:
            proc = await asyncio.create_subprocess_exec(
                "orb",
                "run",
                "-m",
                self._config.vm_machine,
                "bash",
                "-c",
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
            return proc.returncode or 0
        except (OSError, TimeoutError):
            return 1

    async def _check_orb_available(self) -> bool:
        """Check if the orb CLI is available."""

        try:
            proc = await asyncio.create_subprocess_exec(
                "orb",
                "list",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            return proc.returncode == 0
        except (OSError, FileNotFoundError, TimeoutError):
            return False

    async def cleanup(self) -> None:
        """Clean up temp files."""

        if self._config.local_mode:
            for path in (self._local_video_path, Path(f"/tmp/yomi_ffmpeg_{self._match_id}.log")):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            return

        await self._vm_exec(f"rm -f {self._vm_video_path} /tmp/yomi_ffmpeg_{self._match_id}.log")
