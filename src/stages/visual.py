"""Visual stage: Scene building, advanced prompt generation, image generation, thumbnails."""
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import VisualError
from ..core.models import VisualCheckpoint, Scene, ThumbnailVariant, VisualStyle
from ..services.gemini import get_gemini_client
from ..services.pollinations import get_pollinations_client
from ..services.cache import get_image_cache
from ..utils.text import split_sentences, slugify, wrap_text

logger = logging.getLogger(__name__)

# Style definitions matching the old implementation
STYLES = {
    VisualStyle.CINEMATIC: ("dark moody cinematic photoreal film still, muted desaturated colours, "
                            "dramatic low-key lighting, warm lantern glow, fine film grain, "
                            "figures seen from behind or in silhouette or far away, "
                            "no visible faces, no text, no watermark"),
    VisualStyle.MANUSCRIPT: ("medieval illuminated manuscript style illustration, flat gouache colours, "
                             "small detailed figures, ornate decorative border, aged parchment, "
                             "no text, no watermark"),
    VisualStyle.ENGRAVING: ("black and white 19th century etching engraving illustration, "
                            "fine crosshatching, aged paper, detailed, no text, no watermark"),
}

STYLE_ORDER = [VisualStyle.CINEMATIC, VisualStyle.MANUSCRIPT, VisualStyle.CINEMATIC, VisualStyle.ENGRAVING, VisualStyle.MANUSCRIPT]

# Motion prompts for video clips (from old implementation)
MOTION_PROMPTS = {
    VisualStyle.CINEMATIC: "Slow cinematic camera push-in. Candle or firelight flickers softly, fog and dust drift through the air, tiny embers float upwards. Subtle natural movement only, no cuts, no text, no new objects.",
    VisualStyle.MANUSCRIPT: "Gently animate this medieval painting: small figures move slightly, smoke rises, flames flicker, slow camera drift. Keep the painted style exactly, no cuts, no text.",
    VisualStyle.ENGRAVING: "Gently animate this engraving: slight movement of figures and animals, smoke or steam rising, slow camera drift. Keep the black and white etched style exactly, no cuts, no text.",
}


class VisualStage:
    """Visual generation stage handler with advanced prompting."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.gemini = get_gemini_client()
        self.pollinations = get_pollinations_client()
        self.image_cache = get_image_cache()
        
        self.image_seconds = self.config.get('visual', 'image_seconds', default=25)
        self.shot_seconds = self.config.get('visual', 'shot_seconds', default=9)
        self.fps = self.config.get('visual', 'fps', default=24)
        self.resolution = self.config.get('visual', 'resolution', default=[1920, 1080])
        self.broll_every = self.config.get('visual', 'broll_every', default=3)
        self.crop_right = self.config.get('visual', 'crop_right', default=0.15)
        self.crop_bottom = self.config.get('visual', 'crop_bottom', default=0.06)
        
        # Thumbnail settings
        self.thumb_resolution = self.config.get('thumbnails', 'resolution', default=[1280, 720])
        self.thumb_variants = self.config.get('thumbnails', 'variants', default=3)
        self.font_path = self.config.get('thumbnails', 'font_path', 
            default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    
    def build_scenes(self, sentences: List[Dict[str, Any]]) -> List[Scene]:
        """Build scenes from timestamped sentences."""
        scenes = []
        current_scene = []
        scene_start = 0.0
        
        for sent in sentences:
            current_scene.append(sent)
            duration = sent['end'] - scene_start
            
            # Change scene at sentence end if duration threshold met
            # Or at paragraph end if minimum duration met
            should_split = False
            if duration >= self.image_seconds * 0.9:
                should_split = True
            elif sent.get('is_paragraph_end', False) and duration >= self.image_seconds * 0.55:
                should_split = True
            
            if should_split:
                scene_text = " ".join(s['text'] for s in current_scene)
                scenes.append(Scene(
                    index=len(scenes),
                    start=scene_start,
                    end=sent['end'],
                    text=scene_text,
                    prompt="",
                    style=STYLE_ORDER[len(scenes) % len(STYLE_ORDER)]
                ))
                current_scene = []
                scene_start = sent['end']
        
        # Handle remaining
        if current_scene:
            scene_text = " ".join(s['text'] for s in current_scene)
            scenes.append(Scene(
                index=len(scenes),
                start=scene_start,
                end=current_scene[-1]['end'],
                text=scene_text,
                prompt="",
                style=STYLE_ORDER[len(scenes) % len(STYLE_ORDER)]
            ))
        
        # Merge very short final scene
        if len(scenes) > 1 and scenes[-1].end - scenes[-1].start < 8:
            scenes[-2].text += " " + scenes[-1].text
            scenes[-2].end = scenes[-1].end
            scenes.pop()
        
        # Fix scene boundaries
        for i in range(len(scenes) - 1):
            scenes[i].end = scenes[i + 1].start
        if scenes:
            scenes[0].start = 0.0
        
        logger.info(f"Built {len(scenes)} scenes")
        return scenes
    
    def generate_prompts(self, topic: str, scenes: List[Scene]) -> List[Scene]:
        """Generate sophisticated image prompts for each scene."""
        # Check for cached prompts in checkpoint
        visual_checkpoint = self.checkpoint_mgr.load('visual')
        if visual_checkpoint and 'scenes' in visual_checkpoint:
            cached_scenes = visual_checkpoint['scenes']
            if len(cached_scenes) == len(scenes) and all(s.get('prompt') for s in cached_scenes):
                logger.debug("Using cached prompts")
                for i, sc in enumerate(cached_scenes):
                    scenes[i].prompt = sc['prompt']
                    scenes[i].style = VisualStyle(sc['style'])
                return scenes
        
        # Generate prompts in batches
        batch_size = 10
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i:i + batch_size]
            
            listing = "\n".join(f"{k+1}. {s['text'][:500]}" for k, s in enumerate(batch))
            
            prompt = f"""Story: "{topic}" (a calm history documentary for sleep).
Below are numbered passages from the narration. For EACH passage write an image prompt (max 30 words) showing exactly what that passage describes: the place, people, objects, atmosphere, lighting. No meta-commentary. Just the visual description.
Return JSON list of objects with "prompt" and "style" (cinematic/manuscript/engraving).

Passages:
{listing}"""
            
            try:
                response = self.gemini.generate_text(prompt, json_output=True)
                prompts_data = json.loads(response) if isinstance(response, str) else response
                
                for j, pdata in enumerate(prompts_data):
                    idx = i + j
                    if idx < len(scenes):
                        scenes[idx].prompt = pdata.get('prompt', scenes[idx].text[:100])
                        style_str = pdata.get('style', 'cinematic')
                        try:
                            scenes[idx].style = VisualStyle(style_str)
                        except ValueError:
                            scenes[idx].style = VisualStyle.CINEMATIC
                            
            except Exception as e:
                logger.warning(f"Prompt generation failed for batch {i//batch_size}: {e}")
                # Use fallback prompts
                for j, scene in enumerate(batch):
                    idx = i + j
                    scene.prompt = scene.text[:200]
                    scene.style = STYLE_ORDER[idx % len(STYLE_ORDER)]
        
        return scenes
    
    def build_full_prompt(self, scene: Scene) -> str:
        """Build full image generation prompt with style."""
        style_prompt = STYLES.get(scene.style, STYLES[VisualStyle.CINEMATIC])
        return f"{scene.prompt}. {style_prompt}. Wide 16:9 landscape image, cinematic composition, high detail."
    
    def get_motion_prompt(self, style: VisualStyle) -> str:
        """Get motion prompt for video clip generation."""
        return MOTION_PROMPTS.get(style, MOTION_PROMPTS[VisualStyle.CINEMATIC])
    
    def generate_images(self, topic: str, scenes: List[Scene]) -> List[Scene]:
        """Generate images for all scenes using Pollinations."""
        output_dir = self.output_dir / self.topic_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for scene in scenes:
            img_path = output_dir / f"img_{scene.index:03d}.png"
            
            if img_path.exists():
                logger.debug(f"Image exists: {img_path}")
                scene.image_path = str(img_path)
                continue
            
            full_prompt = self.build_full_prompt(scene)
            cache_key = hashlib.sha256(full_prompt.encode()).hexdigest()[:32]
            
            # Check cache
            cached_path = self.image_cache.get_image_path(cache_key)
            if cached_path and cached_path.exists():
                logger.debug(f"Using cached image for scene {scene.index}")
                import shutil
                shutil.copy2(cached_path, img_path)
                scene.image_path = str(img_path)
                continue
            
            logger.info(f"Generating image for scene {scene.index + 1}/{len(scenes)}")
            
            try:
                image_data = self.pollinations.generate_image(
                    full_prompt,
                    width=self.resolution[0],
                    height=self.resolution[1],
                    model="flux",
                    nologo=True,
                    private=True,
                    enhance=True
                )
                
                # Save to output
                with open(img_path, 'wb') as f:
                    f.write(image_data)
                
                # Save to cache
                self.image_cache.save_image(cache_key, image_data, ".png")
                
                scene.image_path = str(img_path)
                logger.info(f"Image saved: {img_path}")
                
            except Exception as e:
                logger.error(f"Image generation failed for scene {scene.index}: {e}")
                # Create placeholder
                import cv2
                import numpy as np
                placeholder = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Scene {scene.index+1}", (100, 540), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)
                cv2.imwrite(str(img_path), placeholder)
                scene.image_path = str(img_path)
        
        return scenes
    
    def generate_thumbnails(self, topic: str, scenes: List[Scene], duration_min: int) -> List[ThumbnailVariant]:
        """Generate thumbnail variants (PIL + optional Gemini)."""
        from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
        import numpy as np
        
        output_dir = self.output_dir / self.topic_slug
        thumbnails = []
        
        if not scenes:
            return thumbnails
        
        # Use images from different parts of the video
        indices = [0, len(scenes)//2, -1] if len(scenes) >= 3 else [0, 0, 0]
        variants = ['a', 'b', 'c']
        
        for idx, variant in zip(indices, variants):
            thumb_path = output_dir / f"thumbnail_{variant}.jpg"
            
            if thumb_path.exists():
                thumbnails.append(ThumbnailVariant(variant=variants.index(variant), path=str(thumb_path)))
                continue
            
            scene = scenes[idx] if idx >= 0 else scenes[len(scenes) + idx]
            
            try:
                # Load and process base image
                im = Image.open(scene.image_path).convert("RGB")
                w, h = im.size
                
                # Crop watermark area
                if scene.image_path.endswith('.jpg') or scene.image_path.endswith('.jpeg'):
                    im = im.crop((0, 0, int(w * (1 - self.crop_right)), int(h * (1 - self.crop_bottom))))
                
                # Resize to thumbnail resolution
                thumb_w, thumb_h = self.thumb_resolution
                im = ImageOps.fit(im, (thumb_w, thumb_h), Image.LANCZOS)
                
                # Enhance
                im = ImageEnhance.Contrast(im).enhance(1.18)
                im = ImageEnhance.Color(im).enhance(1.2)
                im = ImageEnhance.Brightness(im).enhance(1.05)
                
                # Darken edges for text
                arr = np.asarray(im).astype(np.float32)
                xs = np.linspace(0, 1, thumb_w)[None, :]
                ys = np.linspace(0, 1, thumb_h)[:, None]
                
                # Vignette
                vig = np.clip(np.sqrt((xs - 0.5)**2 + (ys - 0.5)**2) * 1.2, 0, 1)**2 * 0.5
                # Side darkening
                if variant == 'a':
                    side = np.clip(1.25 - xs * 1.9, 0, 1)
                elif variant == 'b':
                    side = np.clip(xs * 1.9 - 0.65, 0, 1)
                else:
                    side = np.clip(ys * 1.9 - 0.5, 0, 1)
                
                dark = np.clip(side * 0.78 + vig, 0, 0.9)
                im = Image.fromarray((arr * (1 - dark[..., None])).astype(np.uint8))
                
                # Add text
                draw = ImageDraw.Draw(im)
                
                # Try to load font
                try:
                    font = ImageFont.truetype(self.font_path, 130)
                except:
                    font = ImageFont.load_default()
                
                # Wrap title
                lines = wrap_text(topic.upper(), 14).split('\n')
                line_h = int(130 * 1.12)
                th = line_h * len(lines)
                y = (thumb_h - th) // 2 if variant != 'c' else thumb_h - 70 - th
                
                for k, line in enumerate(lines):
                    x = 60 if variant == 'a' else (thumb_w - 60 - draw.textlength(line, font=font) if variant == 'b' else (thumb_w - draw.textlength(line, font=font)) / 2)
                    color = (255, 214, 120) if k == len(lines) - 1 else (255, 244, 225)
                    draw.text((x, y + k * line_h), line, font=font, fill=color, 
                             stroke_width=7, stroke_fill=(10, 8, 20))
                
                # Add badge
                badge_text = f"{duration_min} MIN SLEEP STORY"
                try:
                    badge_font = ImageFont.truetype(self.font_path, 36)
                except:
                    badge_font = font
                bw = draw.textlength(badge_text, font=badge_font)
                bx = 60 if variant == 'a' else (thumb_w - 60 - bw - 40 if variant == 'b' else (thumb_w - bw - 40) / 2)
                draw.rounded_rectangle((bx, 48, bx + bw + 40, 48 + 62), radius=30, 
                                      fill=(15, 20, 45), outline=(255, 214, 120), width=3)
                draw.text((bx + 20, 56), badge_text, font=badge_font, fill=(255, 255, 255))
                
                im.save(thumb_path, quality=92)
                thumbnails.append(ThumbnailVariant(variant=variants.index(variant), path=str(thumb_path)))
                logger.info(f"Thumbnail {variant} saved: {thumb_path}")
                
            except Exception as e:
                logger.error(f"Thumbnail {variant} failed: {e}")
        
        return thumbnails
    
    def run(
        self,
        timestamps_checkpoint: Dict[str, Any],
        script_checkpoint: Dict[str, Any]
    ) -> VisualCheckpoint:
        """Execute visual stage."""
        logger.info("Starting visual stage")
        
        topic = script_checkpoint.get('topic', self.topic_slug)
        duration_min = int(script_checkpoint.get('estimated_duration_min', 22))
        sentences = timestamps_checkpoint.get('sentences', [])
        
        # Build scenes
        scenes = self.build_scenes(sentences)
        
        # Generate prompts
        scenes = self.generate_prompts(topic, scenes)
        
        # Generate images
        scenes = self.generate_images(topic, scenes)
        
        # Generate thumbnails
        thumbnails = self.generate_thumbnails(topic, scenes, duration_min)
        
        # Also generate motion prompts for broll clips
        for scene in scenes:
            if scene.index % self.broll_every == 0:
                scene.prompt_params = scene.prompt_params or {}
                scene.prompt_params['motion_prompt'] = self.get_motion_prompt(scene.style)
        
        checkpoint = VisualCheckpoint(
            scenes=[s.model_dump() for s in scenes],
            thumbnails=[t.model_dump() for t in thumbnails]
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('visual', checkpoint.model_dump(), success=True)
        
        logger.info(f"Visual complete: {len(scenes)} scenes, {len(thumbnails)} thumbnails")
        return checkpoint


def run_visual(
    output_dir: Path,
    topic_slug: str,
    timestamps_checkpoint: Dict[str, Any],
    script_checkpoint: Dict[str, Any]
) -> VisualCheckpoint:
    """Run visual stage."""
    stage = VisualStage(output_dir, topic_slug)
    return stage.run(timestamps_checkpoint, script_checkpoint)
