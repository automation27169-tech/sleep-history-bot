"""FFmpeg wrapper for audio/video processing."""
import subprocess
import logging
import json
import shlex
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ..core.config import get_config
from ..core.exceptions import FFmpegError

logger = logging.getLogger(__name__)


@dataclass
class AudioInfo:
    """Audio file information."""
    duration: float
    sample_rate: int
    channels: int
    codec: str
    bitrate: int


@dataclass
class VideoInfo:
    """Video file information."""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    audio: Optional[AudioInfo] = None


class FFmpegWrapper:
    """Wrapper for FFmpeg and FFprobe commands."""
    
    def __init__(self):
        self.config = get_config()
        self.ffmpeg = "ffmpeg"
        self.ffprobe = "ffprobe"
    
    def _run(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run FFmpeg command."""
        logger.debug(f"FFmpeg: {' '.join(shlex.quote(c) for c in cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise FFmpegError(f"Command failed: {result.stderr[-1000:]}")
        return result
    
    # ============================================================
    # PROBE FUNCTIONS
    # ============================================================
    
    def get_duration(self, path: Path) -> float:
        """Get media duration in seconds."""
        result = self._run([
            self.ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path)
        ])
        return float(result.stdout.strip())
    
    def get_audio_info(self, path: Path) -> AudioInfo:
        """Get audio stream info."""
        result = self._run([
            self.ffprobe, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,codec_name,bit_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ])
        data = json.loads(result.stdout)
        
        fmt = data.get('format', {})
        stream = data.get('streams', [{}])[0]
        
        return AudioInfo(
            duration=float(fmt.get('duration', 0)),
            sample_rate=int(stream.get('sample_rate', 0)),
            channels=int(stream.get('channels', 0)),
            codec=stream.get('codec_name', ''),
            bitrate=int(stream.get('bit_rate', 0))
        )
    
    def get_video_info(self, path: Path) -> VideoInfo:
        """Get video stream info."""
        result = self._run([
            self.ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name,bit_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ])
        data = json.loads(result.stdout)
        
        fmt = data.get('format', {})
        stream = data.get('streams', [{}])[0]
        
        # Parse frame rate
        fps_str = stream.get('r_frame_rate', '0/1')
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den else 0
        
        audio_info = None
        try:
            audio_info = self.get_audio_info(path)
        except FFmpegError:
            pass
        
        return VideoInfo(
            duration=float(fmt.get('duration', 0)),
            width=int(stream.get('width', 0)),
            height=int(stream.get('height', 0)),
            fps=fps,
            codec=stream.get('codec_name', ''),
            bitrate=int(stream.get('bit_rate', 0)),
            audio=audio_info
        )
    
    # ============================================================
    # AUDIO PROCESSING
    # ============================================================
    
    def concat_audio(self, input_files: List[Path], output_path: Path) -> None:
        """Concatenate audio files."""
        # Create concat file list
        list_path = output_path.with_suffix('.txt')
        with open(list_path, 'w') as f:
            for inp in input_files:
                f.write(f"file '{inp.resolve()}'\n")
        
        self._run([
            self.ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-c", "copy", str(output_path)
        ])
        list_path.unlink(missing_ok=True)
    
    def loudnorm(
        self,
        input_path: Path,
        output_path: Path,
        target_loudness: float = -16,
        target_lra: float = 7,
        target_tp: float = -1.5,
        pad_end: float = 6
    ) -> None:
        """Apply EBU R128 loudness normalization."""
        af = f"loudnorm=I={target_loudness}:LRA={target_lra}:TP={target_tp},apad=pad_dur={pad_end}"
        self._run([
            self.ffmpeg, "-y", "-i", str(input_path),
            "-af", af, str(output_path)
        ])
    
    def convert_audio(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 24000,
        channels: int = 1,
        codec: str = "pcm_s16le"
    ) -> None:
        """Convert audio format."""
        self._run([
            self.ffmpeg, "-y", "-i", str(input_path),
            "-ar", str(sample_rate), "-ac", str(channels),
            "-c:a", codec, str(output_path)
        ])
    
    def encode_mp3(
        self,
        input_path: Path,
        output_path: Path,
        bitrate: str = "96k"
    ) -> None:
        """Encode audio to MP3."""
        self._run([
            self.ffmpeg, "-y", "-i", str(input_path),
            "-ac", "1", "-b:a", bitrate, str(output_path)
        ])
    
    # ============================================================
    # VIDEO PROCESSING
    # ============================================================
    
    def render_video(
        self,
        frames_input: str,  # "-" for stdin, or file pattern
        audio_path: Path,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
        fps: int = 24,
        video_codec: str = "libx264",
        preset: str = "veryfast",
        crf: int = 23,
        maxrate: str = "3500k",
        bufsize: str = "7M",
        audio_codec: str = "aac",
        audio_bitrate: str = "160k",
        pix_fmt: str = "yuv420p",
        colorspace: str = "bt709",
        filter_complex: Optional[str] = None,
        extra_args: Optional[List[str]] = None
    ) -> subprocess.Popen:
        """
        Start video rendering process.
        
        Returns Popen object for streaming frames to stdin.
        """
        cmd = [
            self.ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", frames_input,
            "-i", str(audio_path.resolve())
        ]
        
        if filter_complex:
            cmd += ["-filter_complex", filter_complex]
        
        cmd += [
            "-c:v", video_codec, "-preset", preset, "-crf", str(crf),
            "-maxrate", maxrate, "-bufsize", bufsize,
            "-pix_fmt", pix_fmt,
            "-colorspace", colorspace,
            "-color_primaries", colorspace,
            "-color_trc", colorspace,
            "-c:a", audio_codec, "-b:a", audio_bitrate,
            "-shortest", "-movflags", "+faststart",
            str(output_path.resolve())
        ]
        
        if extra_args:
            cmd += extra_args
        
        logger.info(f"Starting FFmpeg render: {output_path}")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return proc
    
    def extract_frame(self, video_path: Path, output_path: Path, time: float = 0) -> None:
        """Extract a single frame from video."""
        self._run([
            self.ffmpeg, "-y", "-ss", str(time),
            "-i", str(video_path), "-frames:v", "1",
            str(output_path)
        ])
    
    def get_video_duration(self, path: Path) -> float:
        """Get video duration."""
        return self.get_duration(path)
    
    def check_black_frames(
        self,
        video_path: Path,
        threshold: float = 0.1,
        min_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Detect black frames in video."""
        result = self._run([
            self.ffmpeg, "-i", str(video_path),
            "-vf", f"blackdetect=d={min_duration}:pix_th={threshold}",
            "-an", "-f", "null", "-"
        ], check=False)
        
        black_periods = []
        for line in result.stderr.split('\n'):
            if 'blackdetect' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith('black_start:'):
                        start = float(p.split(':')[1])
                    elif p.startswith('black_end:'):
                        end = float(p.split(':')[1])
                        black_periods.append((start, end))
        
        return black_periods
    
    def check_audio_silence(
        self,
        audio_path: Path,
        threshold: float = -50,
        min_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Detect silence in audio."""
        result = self._run([
            self.ffmpeg, "-i", str(audio_path),
            "-af", f"silencedetect=n={threshold}dB:d={min_duration}",
            "-f", "null", "-"
        ], check=False)
        
        silence_periods = []
        for line in result.stderr.split('\n'):
            if 'silence_start' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith('silence_start:'):
                        start = float(p.split(':')[1])
                    elif p.startswith('silence_end:'):
                        end = float(p.split(':')[1])
                        silence_periods.append((start, end))
        
        return silence_periods
    
    def measure_loudness(self, audio_path: Path) -> Dict[str, float]:
        """Measure EBU R128 loudness."""
        result = self._run([
            self.ffmpeg, "-i", str(audio_path),
            "-af", "loudnorm=I=-16:LRA=7:TP=-1.5:print_format=json",
            "-f", "null", "-"
        ], check=False)
        
        # Parse JSON from stderr
        for line in result.stderr.split('\n'):
            if line.strip().startswith('{'):
                try:
                    data = json.loads(line)
                    return {
                        "input_i": data.get("input_i", 0),
                        "input_tp": data.get("input_tp", 0),
                        "input_lra": data.get("input_lra", 0),
                        "input_thresh": data.get("input_thresh", 0),
                        "target_offset": data.get("target_offset", 0)
                    }
                except json.JSONDecodeError:
                    continue
        
        return {}


def get_ffmpeg() -> FFmpegWrapper:
    """Get FFmpeg wrapper instance."""
    return FFmpegWrapper()
