"""Video stage: rendering with Ken Burns, transitions, effects."""
import logging
import time
import subprocess
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import VideoError
from ..core.models import VideoCheckpoint
from ..services.ffmpeg import get_ffmpeg
from ..utils.video import (
    load_image, resize_image, apply_ken_burns, crossfade, 
    apply_vignette, add_film_grain, create_ember_particles,
    update_ember_particles, draw_ember_particles
)

logger = logging.getLogger(__name__)


@dataclass
class Shot:
    """A single shot in the video."""
    index: int
    scene_index: int
    kind: str  # "image" or "video"
    img: Optional[int] = None
    style: str = "cinematic"
    start: float = 0.0
    end: float = 0.0
    a: float = 0.0  # fade in start
    b: float = 0.0  # fade out end
    fi: float = 0.0  # fade in duration
    fo: float = 0.0  # fade out duration
    params: Dict = None
    clip_path: Optional[str] = None


class VideoStage:
    """Video rendering stage handler."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.ffmpeg = get_ffmpeg()
        
        self.fps = self.config.get('visual', 'fps', default=24)
        self.resolution = self.config.get('visual', 'resolution', default=[1920, 1080])
        self.crossfade = self.config.get('visual', 'crossfade_seconds', default=1.2)
        self.cut_fade = self.config.get('visual', 'cut_fade_seconds', default=0.35)
        self.shot_seconds = self.config.get('visual', 'shot_seconds', default=9)
        self.broll_every = self.config.get('visual', 'broll_every', default=3)
        self.embers_enabled = self.config.get('visual', 'embers_enabled', default=True)
        self.three_d_enabled = self.config.get('visual', 'three_d_enabled', default=True)
        
        self.OUT_W, self.OUT_H = self.resolution
        self.FADE = self.crossfade
        self.CUT_FADE = self.cut_fade
        
        # Ember style gains
        self.EMBER_GAIN = {
            "cinematic": 1.0,
            "manuscript": 0.3,
            "engraving": 0.1,
        }
        
        # Load logo
        self.logo = None
        self._load_logo()
        
        # Ember particles
        self.ember_particles = None
        if self.embers_enabled:
            self.ember_particles = create_ember_particles(
                np.zeros((self.OUT_H, self.OUT_W, 3), dtype=np.float32),
                count=80
            )
    
    def _load_logo(self):
        """Load channel logo if exists."""
        logo_path = Path("assets/logo.png")
        if logo_path.exists():
            import cv2
            logo_img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                if logo_img.shape[2] == 4:
                    bgr = logo_img[:, :, :3].astype(np.float32) / 255.0
                    alpha = logo_img[:, :, 3:4].astype(np.float32) / 255.0
                else:
                    bgr = logo_img.astype(np.float32) / 255.0
                    alpha = np.ones((logo_img.shape[0], logo_img.shape[1], 1), dtype=np.float32)
                self.logo = (bgr, alpha)
    
    def build_shots(self, scenes: List[Dict], total_duration: float, broll_clips: List[Dict] = None) -> List[Shot]:
        """Build shot list from scenes."""
        shots = []
        broll_clips = broll_clips or []
        
        for i, scene in enumerate(scenes):
            scene_duration = scene['end'] - scene['start']
            n_shots = max(1, int(scene_duration / self.shot_seconds))
            shot_duration = scene_duration / n_shots
            
            for s in range(n_shots):
                shot_start = scene['start'] + s * shot_duration
                shot_end = shot_start + shot_duration
                
                # Check for broll
                is_broll = (broll_clips and i % self.broll_every == 0 and s == 0 and 
                           (i // self.broll_every) < len(broll_clips))
                
                if is_broll:
                    clip = broll_clips[i // self.broll_every]
                    shots.append(Shot(
                        index=len(shots),
                        scene_index=i,
                        kind="video",
                        clip_path=clip['path'],
                        style=scene['style'],
                        start=shot_start,
                        end=shot_end
                    ))
                else:
                    shots.append(Shot(
                        index=len(shots),
                        scene_index=i,
                        kind="image",
                        img=i,
                        style=scene['style'],
                        start=shot_start,
                        end=shot_end,
                        params=scene.get('prompt_params', {})
                    ))
        
        # Calculate fade times
        n = len(shots)
        for i, shot in enumerate(shots):
            shot.fi = self.FADE if i > 0 else 0.0
            shot.fo = self.FADE if i < n - 1 else 0.0
            shot.a = 0.0 if i == 0 else shot.start - shot.fi / 2
            shot.b = total_duration if i == n - 1 else shots[i + 1].start + shots[i + 1].fi / 2
        
        logger.info(f"Built {len(shots)} shots")
        return shots
    
    def load_scene_image(self, scene_idx: int, scene_data: Dict) -> np.ndarray:
        """Load and prepare scene image."""
        img_path = scene_data.get('image_path')
        if not img_path:
            # Create placeholder
            img = np.zeros((self.OUT_H, self.OUT_W, 3), dtype=np.float32)
            return img
        
        img = load_image(Path(img_path))
        img = resize_image(img, self.OUT_W, self.OUT_H)
        
        return img
    
    def render_frame(self, shot: Shot, progress: float, scene_images: Dict, t: float) -> np.ndarray:
        """Render a single frame for a shot."""
        if shot.kind == "video":
            # Video clip rendering would go here
            # For now, return black frame
            return np.zeros((self.OUT_H, self.OUT_W, 3), dtype=np.float32)
        
        # Image shot with Ken Burns
        img = scene_images[shot.img]
        style = shot.style
        params = shot.params or {}
        
        # Apply Ken Burns
        zoom = params.get('kz', 0)
        if zoom > 0 and self.three_d_enabled:
            # Simple zoom
            frame = apply_ken_burns(img, progress, zoom_start=1.0, zoom_end=1.0 + zoom * 0.2)
        else:
            frame = apply_ken_burns(img, progress, zoom_start=1.0, zoom_end=1.05)
        
        # Apply style effects
        if style == "cinematic":
            frame = apply_vignette(frame, 0.4)
            frame = add_film_grain(frame, 0.015)
        
        # Ember effect
        if self.embers_enabled and self.EMBER_GAIN.get(style, 0) > 0.5 and self.ember_particles:
            self.ember_particles = update_ember_particles(self.ember_particles, 1.0/self.fps, (self.OUT_H, self.OUT_W))
            frame = draw_ember_particles(frame, self.ember_particles)
            # Flicker
            flicker = 1.0 + 0.025 * math.sin(t * 9.0) * math.sin(t * 2.3 + 1.0)
            frame = np.clip(frame * flicker, 0, 1)
        
        # Logo
        if self.logo:
            bgr, alpha = self.logo
            h, w = bgr.shape[:2]
            y1, x1 = self.OUT_H - h - 36, 36
            if y1 >= 0 and x1 >= 0:
                region = frame[y1:y1+h, x1:x1+w].astype(np.float32)
                frame[y1:y1+h, x1:x1+w] = (region * (1 - alpha) + bgr * alpha)
        
        # Fade in/out
        gain = 1.0
        if t < 2.0:
            gain = min(1.0, t / 2.0)
        elif t > shot.b - 4.0:
            gain = min(1.0, max(0.0, (shot.b - t) / 4.0))
        
        if gain < 1.0:
            frame = frame * gain
        
        return np.clip(frame, 0, 1)
    
    def render(
        self,
        visual_checkpoint: Dict[str, Any],
        audio_checkpoint: Dict[str, Any],
        timestamps_checkpoint: Dict[str, Any]
    ) -> VideoCheckpoint:
        """Render final video."""
        logger.info("Starting video render")
        
        scenes = visual_checkpoint['scenes']
        audio_path = Path(audio_checkpoint['narration_path'])
        total_duration = audio_checkpoint['duration_seconds']
        has_subs = bool(timestamps_checkpoint.get('srt_path'))
        
        output_dir = self.output_dir / self.topic_slug
        output_path = output_dir / "video.mp4"
        
        # Load all scene images
        scene_images = {}
        for scene in scenes:
            if scene.get('image_path'):
                scene_images[scene['index']] = self.load_scene_image(scene['index'], scene)
        
        # Build shots
        shots = self.build_shots(scenes, total_duration)
        
        # Prepare filter complex for subtitles
        filter_complex = None
        if has_subs:
            srt_path = timestamps_checkpoint['srt_path']
            filter_complex = f"subtitles={srt_path}:original_size={self.OUT_W}x{self.OUT_H}:force_style='FontName=DejaVu Sans,FontSize=17,PrimaryColour=&H00F0F0F0&,OutlineColour=&H8C000000&,BackColour=&H8C000000&,BorderStyle=3,Outline=3,Shadow=0,MarginV=40,Alignment=2',scale=out_color_matrix=bt709,format=yuv420p"
        else:
            filter_complex = "scale=out_color_matrix=bt709,format=yuv420p"
        
        # Start FFmpeg process
        proc = self.ffmpeg.render_video(
            frames_input="-",
            audio_path=audio_path,
            output_path=output_path,
            width=self.OUT_W,
            height=self.OUT_H,
            fps=self.fps,
            filter_complex=filter_complex
        )
        
        # Render frames
        n_frames = int(total_duration * self.fps)
        t_start = time.time()
        
        try:
            for frame_idx in range(n_frames):
                t = frame_idx / self.fps
                
                # Find current shot
                shot_idx = 0
                while shot_idx + 1 < len(shots) and t >= shots[shot_idx + 1].start:
                    shot_idx += 1
                
                shot = shots[shot_idx]
                
                # Handle crossfade
                if shot_idx + 1 < len(shots):
                    next_shot = shots[shot_idx + 1]
                    if next_shot.fi > 0 and t >= next_shot.start - next_shot.fi / 2:
                        # Crossfade
                        alpha = (t - (next_shot.start - next_shot.fi / 2)) / next_shot.fi
                        frame1 = self.render_frame(shot, 1.0, scene_images, t)
                        frame2 = self.render_frame(next_shot, 0.0, scene_images, t)
                        frame = crossfade(frame1, frame2, alpha)
                    elif shot_idx > 0 and shot.fi > 0 and t < shot.start + shot.fi / 2:
                        # Crossfade from previous
                        prev_shot = shots[shot_idx - 1]
                        alpha = (t - (shot.start - shot.fi / 2)) / shot.fi
                        frame1 = self.render_frame(prev_shot, 1.0, scene_images, t)
                        frame2 = self.render_frame(shot, 0.0, scene_images, t)
                        frame = crossfade(frame1, frame2, alpha)
                    else:
                        frame = self.render_frame(shot, 
                            min(max((t - shot.a) / max(shot.b - shot.a, 1e-6), 0.0), 1.0),
                            scene_images, t)
                else:
                    frame = self.render_frame(shot,
                        min(max((t - shot.a) / max(shot.b - shot.a, 1e-6), 0.0), 1.0),
                        scene_images, t)
                
                # Convert to uint8 BGR
                frame_u8 = (frame * 255).astype(np.uint8)
                proc.stdin.write(frame_u8.tobytes())
                
                # Progress logging
                if frame_idx % (self.fps * 60) == 0 and frame_idx > 0:
                    elapsed = time.time() - t_start
                    pct = 100 * frame_idx / n_frames
                    logger.info(f"Rendering: {pct:.0f}% ({elapsed:.0f}s)")
            
            proc.stdin.close()
            
        except BrokenPipeError:
            pass
        
        # Wait for completion
        stdout, stderr = proc.communicate()
        render_time = time.time() - t_start
        
        if proc.returncode != 0:
            raise VideoError(f"FFmpeg failed: {stderr[-1000:] if stderr else 'Unknown error'}")
        
        # Verify output
        video_info = self.ffmpeg.get_video_info(output_path)
        
        checkpoint = VideoCheckpoint(
            video_path=str(output_path),
            duration_seconds=video_info.duration,
            resolution=[video_info.width, video_info.height],
            fps=int(video_info.fps + 0.5),
            codec="libx264",
            crf=23,
            bitrate_kbps=3500,
            audio_codec="aac",
            audio_bitrate=160,
            has_subtitles=has_subs,
            has_ambience=False,  # Would add ambient audio here
            render_time_seconds=render_time
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('video', checkpoint.model_dump(), success=True)
        
        logger.info(f"Video render complete: {video_info.duration:.1f}s in {render_time:.0f}s")
        return checkpoint


def run_video(
    output_dir: Path,
    topic_slug: str,
    visual_checkpoint: Dict[str, Any],
    audio_checkpoint: Dict[str, Any],
    timestamps_checkpoint: Dict[str, Any]
) -> VideoCheckpoint:
    """Run video stage."""
    stage = VideoStage(output_dir, topic_slug)
    return stage.run(visual_checkpoint, audio_checkpoint, timestamps_checkpoint)
