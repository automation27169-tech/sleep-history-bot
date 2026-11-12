"""Video stage: Advanced rendering with Ken Burns, 3D parallax, embers, transitions."""
import logging
import time
import subprocess
import math
import random
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    params: Dict = field(default_factory=dict)
    clip_path: Optional[str] = None
    clip_duration: float = 0.0
    reader: Any = None  # VideoCapture for clips


@dataclass
class Embers:
    """Ember particle system."""
    particles: List[dict]
    count: int = 80


class VideoStage:
    """Video rendering stage handler with advanced effects."""
    
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
        self.subtitles_enabled = self.config.get('runtime', 'burn_subs', default=False)
        
        self.OUT_W, self.OUT_H = self.resolution
        self.FADE = self.crossfade
        self.CUT_FADE = self.cut_fade
        
        # Ember style gains
        self.EMBER_GAIN = {
            "cinematic": 1.0,
            "manuscript": 0.3,
            "engraving": 0.1,
        }
        
        # Style-specific Ken Burns parameters
        self.STYLE_KB = {
            "cinematic": {"zoom_range": (1.0, 1.15), "pan_range": 0.15},
            "manuscript": {"zoom_range": (1.0, 1.08), "pan_range": 0.08},
            "engraving": {"zoom_range": (1.0, 1.05), "pan_range": 0.05},
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
        
        # Depth model cache
        self._depth_model = None
        self._depth_processor = None
    
    def _load_logo(self):
        """Load channel logo if exists."""
        logo_path = Path("assets/logo.png")
        if logo_path.exists():
            logo_img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                if logo_img.shape[2] == 4:
                    bgr = logo_img[:, :, :3].astype(np.float32) / 255.0
                    alpha = logo_img[:, :, 3:4].astype(np.float32) / 255.0
                else:
                    bgr = logo_img.astype(np.float32) / 255.0
                    alpha = np.ones((logo_img.shape[0], logo_img.shape[1], 1), dtype=np.float32)
                self.logo = (bgr, alpha)
                logger.info(f"Loaded logo: {logo_img.shape[1]}x{logo_img.shape[0]}")
    
    def _get_depth_model(self):
        """Lazy load depth estimation model (MiDaS)."""
        if self._depth_model is None and self.three_d_enabled:
            try:
                logger.info("Loading MiDaS depth model...")
                import torch
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                
                self._depth_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
                self._depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
                self._depth_model.eval()
                logger.info("MiDaS depth model loaded")
            except Exception as e:
                logger.warning(f"Failed to load depth model, 3D disabled: {e}")
                self.three_d_enabled = False
        return self._depth_model, self._depth_processor
    
    def _estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map from image using MiDaS."""
        model, processor = self._get_depth_model()
        if model is None or processor is None:
            return None
        
        try:
            import torch
            # Prepare image
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth = prediction.cpu().numpy()
            
            # Normalize
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth.astype(np.float32)
        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}")
            return None
    
    def _fake_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate fake depth map using gradient."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Use vertical gradient as depth proxy (sky=far, ground=near)
        h, w = gray.shape
        depth = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
        
        # Blend with image gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        
        depth = 0.7 * depth + 0.3 * (1 - grad_mag)
        return np.clip(depth, 0, 1).astype(np.float32)
    
    def build_shots(self, scenes: List[Dict], total_duration: float, broll_clips: List[Dict] = None) -> List[Shot]:
        """Build shot list from scenes with advanced parameters."""
        shots = []
        broll_clips = broll_clips or []
        
        for i, scene in enumerate(scenes):
            scene_duration = scene['end'] - scene['start']
            n_shots = max(1, int(scene_duration / self.shot_seconds))
            shot_duration = scene_duration / n_shots
            
            style = scene.get('style', 'cinematic')
            kb_params = self.STYLE_KB.get(style, self.STYLE_KB['cinematic'])
            
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
                        clip_duration=clip.get('dur', shot_duration),
                        style=style,
                        start=shot_start,
                        end=shot_end
                    ))
                else:
                    # Generate Ken Burns parameters for this shot
                    zoom_start, zoom_end = kb_params["zoom_range"]
                    pan_range = kb_params["pan_range"]
                    pan_x = random.uniform(-pan_range, pan_range)
                    pan_y = random.uniform(-pan_range, pan_range)
                    
                    shots.append(Shot(
                        index=len(shots),
                        scene_index=i,
                        kind="image",
                        img=i,
                        style=style,
                        start=shot_start,
                        end=shot_end,
                        params={
                            'kz': zoom_end - zoom_start,
                            'pan_x': pan_x,
                            'pan_y': pan_y,
                            'zoom_start': zoom_start,
                            'zoom_end': zoom_end,
                        }
                    ))
        
        # Calculate fade times
        n = len(shots)
        for i, shot in enumerate(shots):
            shot.fi = self.FADE if i > 0 else 0.0
            shot.fo = self.FADE if i < n - 1 else 0.0
            shot.a = 0.0 if i == 0 else shot.start - shot.fi / 2
            shot.b = total_duration if i == n - 1 else shots[i + 1].start + shots[i + 1].fi / 2
        
        logger.info(f"Built {len(shots)} shots ({sum(1 for s in shots if s.kind=='video')} video, {sum(1 for s in shots if s.kind=='image')} image)")
        return shots
    
    def load_scene_image(self, scene_idx: int, scene_data: Dict) -> np.ndarray:
        """Load and prepare scene image."""
        img_path = scene_data.get('image_path')
        if not img_path:
            logger.warning(f"No image path for scene {scene_idx}, using placeholder")
            return np.zeros((self.OUT_H, self.OUT_W, 3), dtype=np.float32)
        
        img = load_image(Path(img_path))
        img = resize_image(img, self.OUT_W, self.OUT_H)
        
        # Cache depth map if 3D enabled
        if self.three_d_enabled and scene_data.get('style') == 'cinematic':
            depth = self._estimate_depth(img)
            if depth is None:
                depth = self._fake_depth(img)
            scene_data['_depth_map'] = depth
        
        return img
    
    def render_shot_frame(self, shot: Shot, progress: float, scene_images: Dict, scene_data: Dict, t: float) -> np.ndarray:
        """Render a single frame for a shot."""
        if shot.kind == "video":
            return self._render_video_frame(shot, progress, t)
        
        # Image shot with Ken Burns
        img = scene_images[shot.img]
        style = shot.style
        params = shot.params or {}
        
        # Apply Ken Burns with style-specific params
        zoom_start = params.get('zoom_start', 1.0)
        zoom_end = params.get('zoom_end', 1.05)
        pan_x = params.get('pan_x', 0.0)
        pan_y = params.get('pan_y', 0.0)
        
        frame = apply_ken_burns(img, progress, zoom_start=zoom_start, zoom_end=zoom_end, pan_x=pan_x, pan_y=pan_y)
        
        # Apply 3D parallax for cinematic style
        if style == "cinematic" and params.get('kz', 0) > 0 and self.three_d_enabled:
            depth = scene_data.get('_depth_map')
            if depth is not None:
                frame = self._draw_view_3d(img, depth, progress, params)
        
        # Apply style effects
        if style == "cinematic":
            frame = apply_vignette(frame, 0.4)
            frame = add_film_grain(frame, 0.015)
        elif style == "manuscript":
            frame = apply_vignette(frame, 0.25)
            frame = add_film_grain(frame, 0.01)
        elif style == "engraving":
            frame = apply_vignette(frame, 0.15)
        
        # Ember effect
        if self.embers_enabled and self.EMBER_GAIN.get(style, 0) > 0.5 and self.ember_particles:
            gain = self.EMBER_GAIN[style]
            self.ember_particles = update_ember_particles(self.ember_particles, 1.0/self.fps, (self.OUT_H, self.OUT_W))
            frame = draw_ember_particles(frame, self.ember_particles)
            # Flicker
            flicker = 1.0 + 0.025 * math.sin(t * 9.0) * math.sin(t * 2.3 + 1.0)
            frame = np.clip(frame * flicker, 0, 1)
        
        # Logo overlay
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
    
    def _render_video_frame(self, shot: Shot, progress: float, t: float) -> np.ndarray:
        """Render frame from video clip."""
        if shot.reader is None:
            shot.reader = cv2.VideoCapture(shot.clip_path)
        
        if not shot.reader.isOpened():
            return np.zeros((self.OUT_H, self.OUT_W, 3), dtype=np.float32)
        
        # Calculate frame position in clip
        clip_pos = progress * shot.clip_duration
        shot.reader.set(cv2.CAP_PROP_POS_MSEC, clip_pos * 1000)
        
        ret, frame = shot.reader.read()
        if not ret:
            # Loop or return last frame
            shot.reader.set(cv2.CAP_PROP_POS_MSEC, 0)
            ret, frame = shot.reader.read()
            if not ret:
                return np.zeros((self.OUT_H, self.OUT_W, 3), dtype=np.float32)
        
        # Resize to output resolution
        frame = cv2.resize(frame, (self.OUT_W, self.OUT_H), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def _draw_view_3d(self, image: np.ndarray, depth: np.ndarray, progress: float, params: Dict) -> np.ndarray:
        """Apply 3D parallax effect using depth map."""
        h, w = image.shape[:2]
        
        # Parallax parameters
        zoom_start = params.get('zoom_start', 1.0)
        zoom_end = params.get('zoom_end', 1.15)
        zoom = zoom_start + (zoom_end - zoom_start) * progress
        
        pan_x = params.get('pan_x', 0.0)
        pan_y = params.get('pan_y', 0.0)
        
        # Parallax strength based on depth
        parallax_strength = 0.08  # Maximum pixel shift
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x_coords - w/2) / (w/2)
        y_norm = (y_coords - h/2) / (h/2)
        
        # Apply parallax: closer objects (higher depth value) move more
        # depth is normalized 0-1, where 1 = close, 0 = far
        # Actually in our depth maps, 1=close (foreground), 0=far (background)
        # So we want foreground to move MORE
        shift_x = x_norm * parallax_strength * depth * w * (1 + pan_x)
        shift_y = y_norm * parallax_strength * depth * h * (1 + pan_y)
        
        # Source coordinates
        src_x = x_coords + shift_x
        src_y = y_coords + shift_y
        
        # Apply zoom
        src_x = (src_x - w/2) * zoom + w/2
        src_y = (src_y - h/2) * zoom + h/2
        
        # Remap
        map_x = src_x.astype(np.float32)
        map_y = src_y.astype(np.float32)
        
        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    def _build_ffmpeg_filter(self, has_subs: bool, srt_path: str = "") -> str:
        """Build FFmpeg filter complex."""
        vf = "scale=out_color_matrix=bt709,format=yuv420p"
        
        if has_subs and srt_path:
            sub_style = ("FontName=DejaVu Sans,FontSize=17,PrimaryColour=&H00F0F0F0&,"
                        "OutlineColour=&H8C000000&,BackColour=&H8C000000&,"
                        "BorderStyle=3,Outline=3,Shadow=0,MarginV=40,Alignment=2")
            vf = f"subtitles={srt_path}:original_size={self.OUT_W}x{self.OUT_H}:force_style='{sub_style}',{vf}"
        
        return vf
    
    def run(
        self,
        visual_checkpoint: Dict[str, Any],
        audio_checkpoint: Dict[str, Any],
        timestamps_checkpoint: Dict[str, Any]
    ) -> VideoCheckpoint:
        """Render final video with all effects."""
        logger.info("Starting advanced video render")
        
        scenes = visual_checkpoint['scenes']
        audio_path = Path(audio_checkpoint['narration_path'])
        total_duration = audio_checkpoint['duration_seconds']
        has_subs = self.subtitles_enabled and bool(timestamps_checkpoint.get('srt_path'))
        srt_path = timestamps_checkpoint.get('srt_path', '')
        
        output_dir = self.output_dir / self.topic_slug
        output_path = output_dir / "video.mp4"
        
        # Load all scene images
        scene_images = {}
        for scene in scenes:
            if scene.get('image_path'):
                scene_images[scene['index']] = self.load_scene_image(scene['index'], scene)
        
        # Build shots
        shots = self.build_shots(scenes, total_duration)
        
        # Prepare FFmpeg
        filter_complex = self._build_ffmpeg_filter(has_subs, srt_path)
        
        # Audio setup - check for ambient audio
        ambient_path = output_dir / "ambient.wav"
        use_ambient = ambient_path.exists()
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.OUT_W}x{self.OUT_H}",
            "-framerate", str(self.fps),
            "-i", "-",
            "-i", str(audio_path.resolve())
        ]
        
        if use_ambient:
            cmd += ["-stream_loop", "-1", "-i", str(ambient_path.resolve())]
            af = ("[2:a]lowpass=f=1800,aecho=0.8:0.6:700|1100:0.25|0.15,volume=0.12[m];"
                  "[1:a][m]amix=inputs=2:duration=first:normalize=0[a]")
            cmd += ["-filter_complex", filter_complex + ";" + af, "-map", "[v]", "-map", "[a]"]
        else:
            cmd += ["-filter_complex", filter_complex, "-map", "[v]", "-map", "1:a"]
        
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-maxrate", "3500k", "-bufsize", "7M",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac", "-b:a", "160k",
            "-shortest", "-movflags", "+faststart",
            str(output_path.resolve())
        ]
        
        logger.info(f"Starting FFmpeg: {output_path}")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Render frames
        n_frames = int(total_duration * self.fps)
        t_start = time.time()
        logf = open(output_dir / "ffmpeg.log", "w")
        
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
                        # Crossfade to next
                        alpha = (t - (next_shot.start - next_shot.fi / 2)) / next_shot.fi
                        frame1 = self.render_shot_frame(shot, 1.0, scene_images, scenes[shot.scene_index], t)
                        frame2 = self.render_shot_frame(next_shot, 0.0, scene_images, scenes[next_shot.scene_index], t)
                        frame = crossfade(frame1, frame2, alpha)
                    elif shot_idx > 0 and shot.fi > 0 and t < shot.start + shot.fi / 2:
                        # Crossfade from previous
                        prev_shot = shots[shot_idx - 1]
                        alpha = (t - (shot.start - shot.fi / 2)) / shot.fi
                        frame1 = self.render_shot_frame(prev_shot, 1.0, scene_images, scenes[prev_shot.scene_index], t)
                        frame2 = self.render_shot_frame(shot, 0.0, scene_images, scenes[shot.scene_index], t)
                        frame = crossfade(frame1, frame2, alpha)
                    else:
                        # Normal rendering
                        progress = min(max((t - shot.a) / max(shot.b - shot.a, 1e-6), 0.0), 1.0)
                        frame = self.render_shot_frame(shot, progress, scene_images, scenes[shot.scene_index], t)
                else:
                    # Last shot
                    progress = min(max((t - shot.a) / max(shot.b - shot.a, 1e-6), 0.0), 1.0)
                    frame = self.render_shot_frame(shot, progress, scene_images, scenes[shot.scene_index], t)
                
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
        finally:
            # Cleanup video readers
            for shot in shots:
                if shot.reader:
                    shot.reader.release()
        
        # Wait for completion
        stdout, stderr = proc.communicate()
        render_time = time.time() - t_start
        logf.write(stderr.decode('utf-8', errors='ignore'))
        logf.close()
        
        if proc.returncode != 0:
            err_msg = stderr.decode('utf-8', errors='ignore')[-2000:]
            raise VideoError(f"FFmpeg failed (code {proc.returncode}): {err_msg}")
        
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
            has_ambience=use_ambient,
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
