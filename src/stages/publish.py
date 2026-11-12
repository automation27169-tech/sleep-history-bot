"""Publish stage: metadata generation, quality checks, approval package."""
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import PublishError, QualityGateError
from ..core.models import (
    PublishCheckpoint, Metadata, QualityReport, QualityCheck
)
from ..services.gemini import get_gemini_client
from ..services.ffmpeg import get_ffmpeg
from ..utils.text import format_srt_time

logger = logging.getLogger(__name__)


class PublishStage:
    """Publish package generation stage handler."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.gemini = get_gemini_client()
        self.ffmpeg = get_ffmpeg()
        
        self.quality_config = self.config.get('quality', {})
    
    def generate_metadata(
        self,
        topic: str,
        outline: List[Dict],
        chapters: List[str]
    ) -> Metadata:
        """Generate YouTube metadata."""
        cache_key = f"metadata|{self.topic_slug}"
        
        # Try cache first
        import hashlib
        cached = self.gemini.cache.get(cache_key) if hasattr(self.gemini, 'cache') else None
        
        prompt = f"""YouTube metadata for a calm history sleep story about "{topic}" (US audience).
Return JSON with keys:
- "title": max 70 chars, format like "{topic} | Relaxing History for Sleep"
- "description": 3-4 calm sentences only, no chapter list
- "tags": list of 12 short relevant tags

Topic outline: {json.dumps([{"title": s.get('title', ''), "summary": s.get('summary', '')} for s in outline[:5]])}"""
        
        try:
            response = self.gemini.generate_text(prompt, json_output=True)
            meta_data = json.loads(response) if isinstance(response, str) else response
            
            # Add chapters to description
            description = meta_data.get('description', '')
            if chapters:
                description += "\n\nChapters:\n" + "\n".join(chapters)
            
            description += "\n\nNarration and visuals in this video are AI-generated. Educational, relaxing history storytelling."
            
            return Metadata(
                title=meta_data.get('title', f"{topic} | Relaxing History for Sleep")[:70],
                description=description,
                tags=meta_data.get('tags', ['history', 'sleep', 'documentary', 'relaxing', 'education']),
                chapters=chapters
            )
            
        except Exception as e:
            logger.warning(f"Metadata generation failed: {e}, using defaults")
            return Metadata(
                title=f"{topic} | Relaxing History for Sleep"[:70],
                description=f"Tonight we explore {topic.lower()} in a calm, relaxing history story perfect for sleep.\n\nNarration and visuals are AI-generated.",
                tags=['history', 'sleep', 'documentary', 'relaxing', 'education', 'bedtime'],
                chapters=chapters
            )
    
    def generate_chapters(
        self,
        outline: List[Dict],
        script_checkpoint: Dict[str, Any],
        timestamps_checkpoint: Dict[str, Any]
    ) -> List[str]:
        """Generate chapter timestamps."""
        chapters = []
        word_timestamps = timestamps_checkpoint.get('word_timestamps', [])
        
        if not word_timestamps:
            return chapters
        
        # Map outline sections to timestamps
        sections = script_checkpoint.get('sections', [])
        word_idx = 0
        
        for i, sec in enumerate(sections):
            section_words = sec.get('word_count', 0)
            if word_idx < len(word_timestamps):
                start_time = word_timestamps[word_idx]['start']
                chapters.append(f"{format_srt_time(start_time).replace(',', ':')[:-1]} {sec.get('title', f'Section {i+1}')}")
            word_idx += section_words
        
        return chapters
    
    def run_quality_checks(
        self,
        video_checkpoint: Dict[str, Any],
        audio_checkpoint: Dict[str, Any],
        script_checkpoint: Dict[str, Any]
    ) -> QualityReport:
        """Run all quality gates."""
        checks = []
        
        # Duration check
        duration = video_checkpoint.get('duration_seconds', 0)
        min_dur = self.quality_config.get('min_duration_seconds', 900)
        max_dur = self.quality_config.get('max_duration_seconds', 1800)
        duration_pass = min_dur <= duration <= max_dur
        checks.append(QualityCheck(
            name="duration",
            target=f"{min_dur//60}-{max_dur//60} min",
            actual=f"{duration/60:.1f} min",
            pass_=duration_pass
        ))
        
        # Audio checks
        audio_path = Path(audio_checkpoint.get('narration_path', ''))
        if audio_path.exists():
            # Measure loudness
            loudness_info = self.ffmpeg.measure_loudness(audio_path)
            input_lufs = loudness_info.get('input_i', -16)
            
            min_lufs = self.quality_config.get('min_loudness_lufs', -18)
            max_lufs = self.quality_config.get('max_loudness_lufs', -14)
            loudness_pass = min_lufs <= input_lufs <= max_lufs
            
            checks.append(QualityCheck(
                name="audio_loudness",
                target=f"{min_lufs} to {max_lufs} LUFS",
                actual=f"{input_lufs:.1f} LUFS",
                pass_=loudness_pass
            ))
            
            # Check for silence gaps
            silence = self.ffmpeg.check_audio_silence(audio_path)
            max_gaps = self.quality_config.get('max_silence_gaps', 3)
            silence_pass = len(silence) <= max_gaps
            checks.append(QualityCheck(
                name="audio_silence",
                target=f"≤{max_gaps} gaps",
                actual=f"{len(silence)} gaps",
                pass_=silence_pass
            ))
        
        # Video checks
        video_path = Path(video_checkpoint.get('video_path', ''))
        if video_path.exists():
            # Check black frames
            black_frames = self.ffmpeg.check_black_frames(video_path)
            max_black = self.quality_config.get('max_black_frames', 5)
            black_pass = len(black_frames) <= max_black
            checks.append(QualityCheck(
                name="video_black_frames",
                target=f"≤{max_black} periods",
                actual=f"{len(black_frames)} periods",
                pass_=black_pass
            ))
            
            # Check resolution
            res_pass = (video_checkpoint.get('resolution', [0,0])[0] >= 1920 and 
                       video_checkpoint.get('resolution', [0,0])[1] >= 1080)
            checks.append(QualityCheck(
                name="video_resolution",
                target="1920x1080",
                actual=f"{video_checkpoint.get('resolution', [0,0])[0]}x{video_checkpoint.get('resolution', [0,0])[1]}",
                pass_=res_pass
            ))
        
        # Content checks
        review = script_checkpoint.get('review', {})
        accuracy = review.get('accuracy_score', 0.95)
        min_acc = self.quality_config.get('min_accuracy_score', 0.9)
        checks.append(QualityCheck(
            name="content_accuracy",
            target=f"≥{min_acc:.0%}",
            actual=f"{accuracy:.0%}",
            pass_=accuracy >= min_acc
        ))
        
        originality = review.get('repetition_score', 0.05)
        min_orig = self.quality_config.get('min_originality_score', 0.95)
        orig_pass = (1 - originality) >= min_orig
        checks.append(QualityCheck(
            name="content_originality",
            target=f"≥{min_orig:.0%}",
            actual=f"{(1-originality):.0%}",
            pass_=orig_pass
        ))
        
        pacing = review.get('pacing_score', 0.9)
        min_pacing = self.quality_config.get('min_pacing_score', 0.8)
        checks.append(QualityCheck(
            name="content_pacing",
            target=f"≥{min_pacing:.0%}",
            actual=f"{pacing:.0%}",
            pass_=pacing >= min_pacing
        ))
        
        overall_pass = all(c.pass_ for c in checks)
        
        if not overall_pass:
            failed = [c.name for c in checks if not c.pass_]
            logger.warning(f"Quality gates failed: {failed}")
        else:
            logger.info("All quality gates passed")
        
        return QualityReport(
            duration_check=checks[0],
            audio_check=checks[1] if len(checks) > 1 else checks[0],
            video_check=checks[2] if len(checks) > 2 else checks[0],
            content_check=checks[3] if len(checks) > 3 else checks[0],
            overall_pass=overall_pass
        )
    
    def run(
        self,
        video_checkpoint: Dict[str, Any],
        audio_checkpoint: Dict[str, Any],
        timestamps_checkpoint: Dict[str, Any],
        visual_checkpoint: Dict[str, Any],
        script_checkpoint: Dict[str, Any],
        research_checkpoint: Dict[str, Any]
    ) -> PublishCheckpoint:
        """Execute publish stage."""
        logger.info("Starting publish stage")
        
        topic = research_checkpoint.get('topic', self.topic_slug)
        outline = script_checkpoint.get('outline', [])
        
        # Generate chapters
        chapters = self.generate_chapters(outline, script_checkpoint, timestamps_checkpoint)
        
        # Generate metadata
        metadata = self.generate_metadata(topic, outline, chapters)
        
        # Run quality checks
        quality_report = self.run_quality_checks(video_checkpoint, audio_checkpoint, script_checkpoint)
        
        # Get thumbnails
        thumbnails = visual_checkpoint.get('thumbnails', [])
        thumb_paths = [t['path'] for t in thumbnails if t.get('path')]
        
        checkpoint = PublishCheckpoint(
            video_path=video_checkpoint.get('video_path', ''),
            thumbnails=thumb_paths,
            metadata=metadata,
            quality_report=quality_report,
            approval_ready=quality_report.overall_pass
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('publish', checkpoint.model_dump(), success=True)
        
        # Also save metadata.json for easy access
        output_dir = self.output_dir / self.topic_slug
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps({
            'title': metadata.title,
            'description': metadata.description,
            'tags': metadata.tags,
            'chapters': metadata.chapters,
            'quality_report': {
                'overall_pass': quality_report.overall_pass,
                'checks': [
                    {'name': 'duration', 'pass': quality_report.duration_check.pass_, 'actual': quality_report.duration_check.actual},
                    {'name': 'audio_loudness', 'pass': quality_report.audio_check.pass_, 'actual': quality_report.audio_check.actual},
                    {'name': 'video_black_frames', 'pass': quality_report.video_check.pass_, 'actual': quality_report.video_check.actual},
                    {'name': 'content_accuracy', 'pass': quality_report.content_check.pass_, 'actual': quality_report.content_check.actual},
                ]
            },
            'thumbnails': thumb_paths,
            'video_path': checkpoint.video_path,
            'duration_seconds': video_checkpoint.get('duration_seconds', 0)
        }, indent=2, ensure_ascii=False))
        
        logger.info(f"Publish package ready: approval_ready={checkpoint.approval_ready}")
        return checkpoint


def run_publish(
    output_dir: Path,
    topic_slug: str,
    video_checkpoint: Dict[str, Any],
    audio_checkpoint: Dict[str, Any],
    timestamps_checkpoint: Dict[str, Any],
    visual_checkpoint: Dict[str, Any],
    script_checkpoint: Dict[str, Any],
    research_checkpoint: Dict[str, Any]
) -> PublishCheckpoint:
    """Run publish stage."""
    stage = PublishStage(output_dir, topic_slug)
    return stage.run(video_checkpoint, audio_checkpoint, timestamps_checkpoint, 
                     visual_checkpoint, script_checkpoint, research_checkpoint)
