"""Video processing utilities."""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class VideoFrame:
    """Video frame with metadata."""
    data: np.ndarray  # BGR format
    timestamp: float
    frame_number: int


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR float32 [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    # Handle alpha channel
    if img.shape[2] == 4:
        # Alpha blend onto black
        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha
    
    return img


def resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to fit within dimensions, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center on canvas
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def apply_ken_burns(
    img: np.ndarray,
    progress: float,  # 0.0 to 1.0
    zoom_start: float = 1.0,
    zoom_end: float = 1.15,
    pan_x: float = 0.0,  # -1 to 1
    pan_y: float = 0.0
) -> np.ndarray:
    """Apply Ken Burns effect (zoom + pan)."""
    h, w = img.shape[:2]
    
    # Interpolate zoom
    zoom = zoom_start + (zoom_end - zoom_start) * progress
    
    # Calculate crop region
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    
    # Center point with pan
    cx = w // 2 + int(pan_x * (w - crop_w) * 0.5)
    cy = h // 2 + int(pan_y * (h - crop_h) * 0.5)
    
    # Clamp
    cx = max(crop_w // 2, min(w - crop_w // 2, cx))
    cy = max(crop_h // 2, min(h - crop_h // 2, cy))
    
    # Crop and resize
    x1, y1 = cx - crop_w // 2, cy - crop_h // 2
    x2, y2 = x1 + crop_w, y1 + crop_h
    
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def crossfade(frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
    """Crossfade between two frames."""
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


def apply_vignette(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply vignette effect."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    max_dist = np.sqrt(cx**2 + cy**2)
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    vignette = 1 - strength * (dist / max_dist)
    vignette = np.clip(vignette, 0, 1)
    
    return img * vignette[..., np.newaxis]


def add_film_grain(img: np.ndarray, strength: float = 0.02) -> np.ndarray:
    """Add film grain noise."""
    noise = np.random.normal(0, strength, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)


def create_ember_particles(
    frame: np.ndarray,
    count: int = 50,
    color: Tuple[float, float, float] = (1.0, 0.6, 0.2),
    lifetime: float = 3.0
) -> List[dict]:
    """Create ember particle data."""
    h, w = frame.shape[:2]
    particles = []
    
    for _ in range(count):
        particles.append({
            'x': np.random.uniform(0, w),
            'y': np.random.uniform(h * 0.5, h),
            'vx': np.random.uniform(-10, 10),
            'vy': np.random.uniform(-50, -20),
            'size': np.random.uniform(1, 3),
            'alpha': np.random.uniform(0.3, 0.8),
            'life': np.random.uniform(0, lifetime),
            'max_life': lifetime,
            'color': color
        })
    
    return particles


def update_ember_particles(particles: List[dict], dt: float, frame_shape: Tuple[int, int]) -> List[dict]:
    """Update ember particle positions."""
    h, w = frame_shape[:2]
    updated = []
    
    for p in particles:
        p['life'] += dt
        if p['life'] >= p['max_life']:
            # Respawn at bottom
            p['x'] = np.random.uniform(0, w)
            p['y'] = h + np.random.uniform(0, 50)
            p['life'] = 0
            p['max_life'] = np.random.uniform(2, 4)
            p['vy'] = np.random.uniform(-60, -30)
        
        p['x'] += p['vx'] * dt
        p['y'] += p['vy'] * dt
        
        # Fade out
        p['alpha'] = max(0, 1 - p['life'] / p['max_life'])
        
        if p['alpha'] > 0:
            updated.append(p)
    
    return updated


def draw_ember_particles(frame: np.ndarray, particles: List[dict]) -> np.ndarray:
    """Draw ember particles on frame."""
    for p in particles:
        x, y = int(p['x']), int(p['y'])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            size = max(1, int(p['size']))
            alpha = p['alpha']
            color = tuple(int(c * 255) for c in p['color'])
            cv2.circle(frame, (x, y), size, color, -1, lineType=cv2.LINE_AA)
    
    return frame
