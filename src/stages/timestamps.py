"""Timestamps stage: Whisper alignment, SRT generation."""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import TimestampsError
from ..core.models import TimestampsCheckpoint, WordTimestamp, SentenceTimestamp
from ..services.whisper import get_whisper_aligner
from ..services.cache import get_cache
from ..utils.text import split_sentences, format_srt_time, wrap_text

logger = logging.getLogger(__name__)


class TimestampsStage:
    """Timestamp alignment stage handler."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.whisper = get_whisper_aligner()
        self.cache = get_cache()
    
    def align_script_to_audio(
        self,
        script_text: str,
        audio_path: Path
    ) -> Dict[str, Any]:
        """Align script words to audio timestamps."""
        cache_key = f"alignment|{self.topic_slug}|{hash(script_text[:100])}"
        
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Using cached alignment")
            return cached
        
        logger.info("Aligning script to audio with Whisper...")
        alignment = self.whisper.align_script_to_audio(script_text, audio_path)
        
        self.cache.set(cache_key, alignment)
        return alignment
    
    def generate_srt(
        self,
        sentences: List[Dict[str, Any]],
        output_path: Path,
        max_chars: int = 84
    ) -> None:
        """Generate SRT subtitle file."""
        cues = []
        
        for sent in sentences:
            text = sent['text']
            start = sent['start']
            end = sent['end']
            
            # Wrap long lines
            if len(text) > max_chars:
                wrapped = wrap_text(text, max_chars // 2)
                text = wrapped
            
            cues.append((start, end, text))
        
        # Fix overlapping cues
        fixed_cues = []
        for i, (start, end, text) in enumerate(cues):
            # Add small buffer
            end = max(end + 0.2, start + 0.6)
            if i + 1 < len(cues):
                end = min(end, cues[i + 1][0])
            fixed_cues.append((start, max(end, start + 0.3), text))
        
        # Write SRT
        srt_lines = []
        for i, (start, end, text) in enumerate(fixed_cues, 1):
            srt_lines.append(f"{i}")
            srt_lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
            srt_lines.append(text)
            srt_lines.append("")
        
        output_path.write_text("\n".join(srt_lines))
        logger.info(f"SRT generated: {output_path} ({len(fixed_cues)} cues)")
    
    def run(
        self,
        script_checkpoint: Dict[str, Any],
        audio_checkpoint: Dict[str, Any]
    ) -> TimestampsCheckpoint:
        """Execute timestamps stage."""
        logger.info("Starting timestamps stage")
        
        script_text = script_checkpoint['full_script']
        audio_path = Path(audio_checkpoint['narration_path'])
        output_dir = self.output_dir / self.topic_slug
        
        # Align script to audio
        alignment = self.align_script_to_audio(script_text, audio_path)
        
        # Convert to models
        word_timestamps = [
            WordTimestamp(word=w['word'], start=w['start'], end=w['end'])
            for w in alignment['word_timestamps']
        ]
        
        sentences = [
            SentenceTimestamp(
                text=s['text'],
                start=s['start'],
                end=s['end'],
                word_start_idx=s['word_start_idx'],
                word_end_idx=s['word_end_idx']
            )
            for s in alignment['sentences']
        ]
        
        # Generate SRT
        srt_path = output_dir / "subs.srt"
        self.generate_srt(alignment['sentences'], srt_path)
        
        checkpoint = TimestampsCheckpoint(
            word_timestamps=word_timestamps,
            sentences=sentences,
            srt_path=str(srt_path),
            alignment_method=alignment['alignment_method'],
            alignment_quality=alignment['alignment_quality']
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('timestamps', checkpoint.model_dump(), success=True)
        
        logger.info(f"Timestamps complete: {len(word_timestamps)} words, {len(sentences)} sentences")
        return checkpoint


def run_timestamps(
    output_dir: Path,
    topic_slug: str,
    script_checkpoint: Dict[str, Any],
    audio_checkpoint: Dict[str, Any]
) -> TimestampsCheckpoint:
    """Run timestamps stage."""
    stage = TimestampsStage(output_dir, topic_slug)
    return stage.run(script_checkpoint, audio_checkpoint)
