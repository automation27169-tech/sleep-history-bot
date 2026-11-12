"""Quality gates for pipeline validation."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from ..core.config import get_config
from ..core.exceptions import QualityGateError
from ..services.ffmpeg import get_ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any]


class QualityGates:
    """Quality gate validator for pipeline outputs."""
    
    def __init__(self):
        self.config = get_config()
        self.quality_config = self.config.get('quality', {})
        self.ffmpeg = get_ffmpeg()
    
    def check_duration(self, duration_seconds: float) -> GateResult:
        """Check video duration is within bounds."""
        min_dur = self.quality_config.get('min_duration_seconds', 900)
        max_dur = self.quality_config.get('max_duration_seconds', 1800)
        
        passed = min_dur <= duration_seconds <= max_dur
        
        return GateResult(
            name="duration",
            passed=passed,
            message=f"Duration {duration_seconds/60:.1f} min ({'OK' if passed else 'OUT OF RANGE'})",
            details={'actual': duration_seconds, 'min': min_dur, 'max': max_dur}
        )
    
    def check_audio_loudness(self, audio_path: Path) -> GateResult:
        """Check audio loudness is within EBU R128 range."""
        try:
            loudness = self.ffmpeg.measure_loudness(audio_path)
            input_lufs = loudness.get('input_i', -16)
            
            min_lufs = self.quality_config.get('min_loudness_lufs', -18)
            max_lufs = self.quality_config.get('max_loudness_lufs', -14)
            
            passed = min_lufs <= input_lufs <= max_lufs
            
            return GateResult(
                name="audio_loudness",
                passed=passed,
                message=f"Loudness {input_lufs:.1f} LUFS ({'OK' if passed else 'OUT OF RANGE'})",
                details={'actual': input_lufs, 'min': min_lufs, 'max': max_lufs}
            )
        except Exception as e:
            return GateResult(
                name="audio_loudness",
                passed=False,
                message=f"Failed to measure loudness: {e}",
                details={'error': str(e)}
            )
    
    def check_audio_silence(self, audio_path: Path) -> GateResult:
        """Check for excessive silence gaps."""
        try:
            silence = self.ffmpeg.check_audio_silence(audio_path)
            max_gaps = self.quality_config.get('max_silence_gaps', 3)
            
            passed = len(silence) <= max_gaps
            
            return GateResult(
                name="audio_silence",
                passed=passed,
                message=f"Silence gaps: {len(silence)} ({'OK' if passed else 'TOO MANY'})",
                details={'count': len(silence), 'max': max_gaps, 'gaps': silence}
            )
        except Exception as e:
            return GateResult(
                name="audio_silence",
                passed=False,
                message=f"Failed to check silence: {e}",
                details={'error': str(e)}
            )
    
    def check_video_black_frames(self, video_path: Path) -> GateResult:
        """Check for black frames in video."""
        try:
            black = self.ffmpeg.check_black_frames(video_path)
            max_black = self.quality_config.get('max_black_frames', 5)
            
            passed = len(black) <= max_black
            
            return GateResult(
                name="video_black_frames",
                passed=passed,
                message=f"Black frame periods: {len(black)} ({'OK' if passed else 'TOO MANY'})",
                details={'count': len(black), 'max': max_black, 'periods': black}
            )
        except Exception as e:
            return GateResult(
                name="video_black_frames",
                passed=False,
                message=f"Failed to check black frames: {e}",
                details={'error': str(e)}
            )
    
    def check_video_resolution(self, video_path: Path, expected: Tuple[int, int] = (1920, 1080)) -> GateResult:
        """Check video resolution."""
        try:
            info = self.ffmpeg.get_video_info(video_path)
            
            passed = info.width >= expected[0] and info.height >= expected[1]
            
            return GateResult(
                name="video_resolution",
                passed=passed,
                message=f"Resolution: {info.width}x{info.height} ({'OK' if passed else 'TOO LOW'})",
                details={'actual': [info.width, info.height], 'expected': list(expected)}
            )
        except Exception as e:
            return GateResult(
                name="video_resolution",
                passed=False,
                message=f"Failed to check resolution: {e}",
                details={'error': str(e)}
            )
    
    def check_audio_video_sync(self, video_path: Path, audio_path: Path) -> GateResult:
        """Check audio/video duration match."""
        try:
            video_dur = self.ffmpeg.get_video_duration(video_path)
            audio_dur = self.ffmpeg.get_duration(audio_path)
            
            diff = abs(video_dur - audio_dur)
            passed = diff < 1.0  # Within 1 second
            
            return GateResult(
                name="av_sync",
                passed=passed,
                message=f"Duration diff: {diff:.2f}s ({'OK' if passed else 'OUT OF SYNC'})",
                details={'video': video_dur, 'audio': audio_dur, 'diff': diff}
            )
        except Exception as e:
            return GateResult(
                name="av_sync",
                passed=False,
                message=f"Failed to check sync: {e}",
                details={'error': str(e)}
            )
    
    def check_content_quality(self, script_checkpoint: Dict[str, Any]) -> GateResult:
        """Check script/content quality metrics."""
        review = script_checkpoint.get('review', {})
        
        accuracy = review.get('accuracy_score', 0.95)
        min_acc = self.quality_config.get('min_accuracy_score', 0.9)
        
        originality = 1 - review.get('repetition_score', 0.05)
        min_orig = self.quality_config.get('min_originality_score', 0.95)
        
        pacing = review.get('pacing_score', 0.9)
        min_pacing = self.quality_config.get('min_pacing_score', 0.8)
        
        passed = (accuracy >= min_acc and originality >= min_orig and pacing >= min_pacing)
        
        return GateResult(
            name="content_quality",
            passed=passed,
            message=f"Accuracy: {accuracy:.0%}, Originality: {originality:.0%}, Pacing: {pacing:.0%}",
            details={
                'accuracy': accuracy, 'min_accuracy': min_acc,
                'originality': originality, 'min_originality': min_orig,
                'pacing': pacing, 'min_pacing': min_pacing
            }
        )
    
    def run_all_gates(
        self,
        video_checkpoint: Dict[str, Any],
        audio_checkpoint: Dict[str, Any],
        script_checkpoint: Dict[str, Any]
    ) -> List[GateResult]:
        """Run all quality gates."""
        results = []
        
        # Duration
        results.append(self.check_duration(video_checkpoint.get('duration_seconds', 0)))
        
        # Audio
        audio_path = Path(audio_checkpoint.get('narration_path', ''))
        if audio_path.exists():
            results.append(self.check_audio_loudness(audio_path))
            results.append(self.check_audio_silence(audio_path))
        
        # Video
        video_path = Path(video_checkpoint.get('video_path', ''))
        if video_path.exists():
            results.append(self.check_video_black_frames(video_path))
            results.append(self.check_video_resolution(video_path))
            if audio_path.exists():
                results.append(self.check_audio_video_sync(video_path, audio_path))
        
        # Content
        results.append(self.check_content_quality(script_checkpoint))
        
        # Log results
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            logger.info(f"Quality gate [{status}]: {r.name} - {r.message}")
        
        return results
    
    def assert_all_pass(self, results: List[GateResult]) -> None:
        """Raise exception if any gate fails."""
        failed = [r for r in results if not r.passed]
        if failed:
            raise QualityGateError(
                gate=failed[0].name,
                message=f"{len(failed)} quality gate(s) failed",
                details={r.name: r.details for r in failed}
            )


def run_quality_gates(
    video_checkpoint: Dict[str, Any],
    audio_checkpoint: Dict[str, Any],
    script_checkpoint: Dict[str, Any]
) -> List[GateResult]:
    """Convenience function to run all quality gates."""
    gates = QualityGates()
    return gates.run_all_gates(video_checkpoint, audio_checkpoint, script_checkpoint)
