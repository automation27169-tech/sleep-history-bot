"""Faster-Whisper wrapper for word-level timestamp alignment."""
import logging
import json
import hashlib
import difflib
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..core.config import get_config
from ..core.exceptions import WhisperError
from ..services.cache import get_cache

logger = logging.getLogger(__name__)


@dataclass
class WhisperWord:
    """Word with timestamps from Whisper."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class AlignedWord:
    """Script word aligned to audio timestamp."""
    word: str
    start: float
    end: float
    matched: bool = False


class WhisperAligner:
    """Aligns script words to audio using Faster-Whisper."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = get_cache()
        self.model_name = self.config.get('timestamps', 'whisper_model', default="base.en")
        self.device = self.config.get('timestamps', 'whisper_device', default="cpu")
        self.compute_type = self.config.get('timestamps', 'whisper_compute_type', default="int8")
        self.min_alignment_ratio = self.config.get('timestamps', 'min_alignment_ratio', default=0.6)
        self.fallback_to_estimated = self.config.get('timestamps', 'fallback_to_estimated', default=True)
        
        self._model = None
    
    @property
    def model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        return self._model
    
    def _normalize_word(self, word: str) -> str:
        """Normalize word for matching."""
        return re.sub(r'[^a-z0-9]', '', word.lower())
    
    def transcribe(self, audio_path: Path) -> List[WhisperWord]:
        """Transcribe audio and return word-level timestamps."""
        cache_key = f"whisper|{self.model_name}|{hashlib.sha256(str(audio_path).encode()).hexdigest()[:16]}"
        
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for Whisper transcription")
            return [WhisperWord(**w) for w in cached]
        
        logger.info(f"Transcribing audio with Whisper: {audio_path}")
        segments, _ = self.model.transcribe(
            str(audio_path),
            language="en",
            word_timestamps=True,
            beam_size=1,
            condition_on_previous_text=False
        )
        
        words = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    words.append(WhisperWord(
                        word=w.word.strip(),
                        start=float(w.start),
                        end=float(w.end),
                        probability=getattr(w, 'probability', 1.0)
                    ))
        
        # Cache result
        self.cache.set(cache_key, [w.__dict__ for w in words])
        
        logger.info(f"Whisper found {len(words)} words")
        return words
    
    def align_words(
        self,
        script_words: List[str],
        whisper_words: List[WhisperWord],
        total_duration: float
    ) -> Tuple[List[float], List[float]]:
        """
        Align script words to whisper words using sequence matching.
        
        Returns (start_times, end_times) for each script word.
        """
        if not script_words:
            return [], []
        
        if not whisper_words:
            logger.warning("No whisper words, using estimated timing")
            return self._estimated_timing(script_words, total_duration)
        
        # Normalize words for matching
        script_norm = [self._normalize_word(w) or f"_s{i}" for i, w in enumerate(script_words)]
        whisper_norm = [self._normalize_word(w.word) or f"_w{i}" for i, w in enumerate(whisper_words)]
        
        # Sequence matching
        matcher = difflib.SequenceMatcher(None, script_norm, whisper_norm, autojunk=False)
        matches = matcher.get_matching_blocks()
        
        n = len(script_words)
        starts = [None] * n
        ends = [None] * n
        matched_count = 0
        
        for match in matches:
            if match.size == 0:
                continue
            for k in range(match.size):
                script_idx = match.a + k
                whisper_idx = match.b + k
                if script_idx < n and whisper_idx < len(whisper_words):
                    starts[script_idx] = whisper_words[whisper_idx].start
                    ends[script_idx] = whisper_words[whisper_idx].end
                    matched_count += 1
        
        alignment_ratio = matched_count / n if n > 0 else 0
        logger.info(f"Word alignment: {matched_count}/{n} ({alignment_ratio:.1%}) matched")
        
        if alignment_ratio < self.min_alignment_ratio:
            logger.warning(f"Alignment ratio {alignment_ratio:.1%} below threshold {self.min_alignment_ratio}")
            if self.fallback_to_estimated:
                return self._estimated_timing(script_words, total_duration)
            raise WhisperError(f"Alignment too poor: {alignment_ratio:.1%}")
        
        # Fill in gaps using interpolation
        starts, ends = self._interpolate_gaps(starts, ends, total_duration)
        
        return starts, ends
    
    def _interpolate_gaps(
        self,
        starts: List[Optional[float]],
        ends: List[Optional[float]],
        total_duration: float
    ) -> Tuple[List[float], List[float]]:
        """Interpolate missing timestamps."""
        n = len(starts)
        speech_end = total_duration - 6.0  # Account for padding
        
        # Forward fill then backward fill for leading/trailing gaps
        last_known = 0.0
        for i in range(n):
            if starts[i] is not None:
                last_known = ends[i]
            elif i > 0:
                starts[i] = last_known
                ends[i] = last_known + 0.3  # Default word duration
        
        # Handle trailing gaps
        next_known = speech_end
        for i in range(n - 1, -1, -1):
            if starts[i] is not None:
                next_known = starts[i]
            elif ends[i] is None or ends[i] >= next_known:
                ends[i] = next_known
                starts[i] = max(0.0, next_known - 0.3)
        
        # Interpolate internal gaps
        i = 0
        while i < n:
            if starts[i] is not None:
                i += 1
                continue
            
            # Find next known
            j = i
            while j < n and starts[j] is None:
                j += 1
            
            if j < n:
                t1 = starts[j]
            else:
                t1 = min(speech_end, (starts[i-1] if i > 0 else 0) + 0.4 * (j - i))
            
            t0 = ends[i-1] if i > 0 else 0.0
            gap_count = j - i
            
            for k in range(gap_count):
                idx = i + k
                progress = (k + 1) / (gap_count + 1)
                starts[idx] = t0 + (t1 - t0) * progress - 0.15
                ends[idx] = t0 + (t1 - t0) * progress + 0.15
                # Ensure monotonic
                if idx > 0 and starts[idx] < ends[idx-1]:
                    starts[idx] = ends[idx-1]
                if ends[idx] < starts[idx] + 0.05:
                    ends[idx] = starts[idx] + 0.05
            
            i = j
        
        # Final cleanup
        for i in range(n):
            if starts[i] is None:
                starts[i] = 0.0
            if ends[i] is None:
                ends[i] = starts[i] + 0.3
            if ends[i] <= starts[i]:
                ends[i] = starts[i] + 0.1
        
        return starts, ends
    
    def _estimated_timing(self, script_words: List[str], total_duration: float) -> Tuple[List[float], List[float]]:
        """Generate estimated timing based on word count and total duration."""
        speech_duration = max(total_duration - 6.0, 1.0)  # Subtract padding
        n = len(script_words)
        
        starts = [speech_duration * i / n for i in range(n)]
        ends = [speech_duration * (i + 1) / n for i in range(n)]
        
        logger.info(f"Using estimated timing for {n} words over {speech_duration:.1f}s")
        return starts, ends
    
    def align_script_to_audio(
        self,
        script_text: str,
        audio_path: Path
    ) -> Dict[str, Any]:
        """
        Full pipeline: transcribe audio, split script, align words.
        
        Returns dict with word_timestamps, sentence_timestamps, and alignment info.
        """
        # Get audio duration
        import subprocess
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(audio_path)
        ], capture_output=True, text=True)
        total_duration = float(result.stdout.strip())
        
        # Transcribe
        whisper_words = self.transcribe(audio_path)
        
        # Split script into words
        script_words = script_text.split()
        
        # Align
        starts, ends = self.align_words(script_words, whisper_words, total_duration)
        
        # Build word timestamps
        word_timestamps = [
            {"word": w, "start": s, "end": e}
            for w, s, e in zip(script_words, starts, ends)
        ]
        
        # Split into sentences for SRT
        sentences = self._split_sentences(script_text, script_words, starts, ends)
        
        return {
            "word_timestamps": word_timestamps,
            "sentences": sentences,
            "alignment_method": "whisper" if whisper_words else "estimated",
            "alignment_quality": len([w for w in word_timestamps if w["start"] > 0]) / len(script_words) if script_words else 0,
            "total_duration": total_duration
        }
    
    def _split_sentences(
        self,
        script_text: str,
        script_words: List[str],
        word_starts: List[float],
        word_ends: List[float]
    ) -> List[Dict[str, Any]]:
        """Split script into sentences with timestamps."""
        sentences = []
        
        # Split by paragraph first
        paragraphs = [p.strip() for p in script_text.split("\n\n") if p.strip()]
        word_idx = 0
        
        for para in paragraphs:
            # Split paragraph into sentences
            # Simple split on .!? followed by space
            sent_texts = re.split(r'(?<=[.!?])\s+', para)
            
            for sent_text in sent_texts:
                sent_text = sent_text.strip()
                if not sent_text:
                    continue
                
                sent_words = sent_text.split()
                if not sent_words:
                    continue
                
                w0 = word_idx
                w1 = word_idx + len(sent_words)
                
                if w1 <= len(script_words):
                    sentences.append({
                        "text": sent_text,
                        "start": word_starts[w0],
                        "end": word_ends[w1 - 1],
                        "word_start_idx": w0,
                        "word_end_idx": w1
                    })
                
                word_idx = w1
        
        return sentences


def get_whisper_aligner() -> WhisperAligner:
    """Get Whisper aligner instance."""
    return WhisperAligner()
