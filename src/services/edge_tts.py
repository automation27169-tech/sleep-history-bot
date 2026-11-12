"""Microsoft Edge TTS wrapper (free, local)."""
import asyncio
import logging
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from ..core.config import get_config
from ..core.exceptions import EdgeTTSError, RetryExhaustedError
from ..services.cache import get_cache

logger = logging.getLogger(__name__)


@dataclass
class EdgeVoice:
    """Edge TTS voice information."""
    name: str
    gender: str
    locale: str
    friendly_name: str


# Recommended voices for sleep narration
RECOMMENDED_VOICES = [
    EdgeVoice("en-US-GuyNeural", "Male", "en-US", "Guy (US)"),
    EdgeVoice("en-US-ChristopherNeural", "Male", "en-US", "Christopher (US)"),
    EdgeVoice("en-US-EricNeural", "Male", "en-US", "Eric (US)"),
    EdgeVoice("en-GB-RyanNeural", "Male", "en-GB", "Ryan (UK)"),
    EdgeVoice("en-AU-WilliamNeural", "Male", "en-AU", "William (AU)"),
    EdgeVoice("en-US-JennyNeural", "Female", "en-US", "Jenny (US)"),
    EdgeVoice("en-US-AriaNeural", "Female", "en-US", "Aria (US)"),
]


class EdgeTTSClient:
    """Wrapper for Microsoft Edge TTS (via edge-tts library)."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = get_cache()
        self.default_voice = self.config.get('audio', 'edge_voice', default="en-US-GuyNeural")
        self.default_rate = "-15%"  # Slower for sleep narration
        self.default_pitch = "-5Hz"  # Slightly lower pitch
        
        # Retry settings
        self.max_retries = self.config.get('pipeline', 'retry_attempts', default=3)
        self.base_delay = self.config.get('pipeline', 'retry_base_delay', default=10)
        self.max_delay = self.config.get('pipeline', 'max_retry_delay', default=60)
    
    def _run_edge_tts(self, text: str, voice: str, rate: str, pitch: str, output_path: Path) -> None:
        """Run edge-tts command."""
        # Use asyncio to run the communicate function
        import edge_tts
        
        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(str(output_path))
        
        asyncio.run(_generate())
    
    def _convert_to_wav(self, input_path: Path, output_path: Path, sample_rate: int = 24000) -> None:
        """Convert audio to WAV using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(sample_rate), "-ac", "1",
            "-c:a", "pcm_s16le", str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise EdgeTTSError(f"FFmpeg conversion failed: {result.stderr}")
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        sample_rate: int = 24000,
        use_cache: bool = True
    ) -> bytes:
        """
        Synthesize text to speech using Edge TTS.
        
        Returns raw WAV bytes (16-bit PCM, mono, specified sample rate).
        """
        voice = voice or self.default_voice
        rate = rate or self.default_rate
        pitch = pitch or self.default_pitch
        
        cache_key = f"edge_tts|{voice}|{rate}|{pitch}|{sample_rate}|{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for Edge TTS")
                return bytes(cached)
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    mp3_path = tmpdir / "output.mp3"
                    wav_path = tmpdir / "output.wav"
                    
                    # Generate MP3
                    self._run_edge_tts(text, voice, rate, pitch, mp3_path)
                    
                    # Convert to WAV
                    self._convert_to_wav(mp3_path, wav_path, sample_rate)
                    
                    # Read WAV data
                    with open(wav_path, 'rb') as f:
                        wav_data = f.read()
                    
                    if use_cache:
                        self.cache.set(cache_key, list(wav_data))
                    
                    logger.debug(f"Edge TTS generated: {len(wav_data)} bytes")
                    return wav_data
                    
            except Exception as e:
                last_error = EdgeTTSError(f"Attempt {attempt + 1} failed: {e}")
                logger.warning(f"Edge TTS attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    import time
                    wait = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(wait)
        
        raise RetryExhaustedError("edge_tts", self.max_retries, last_error)
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        sample_rate: int = 24000
    ) -> None:
        """Synthesize text directly to a WAV file."""
        wav_data = self.synthesize(text, voice, rate, pitch, sample_rate, use_cache=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(wav_data)
    
    @staticmethod
    def list_voices() -> List[EdgeVoice]:
        """List available Edge TTS voices."""
        return RECOMMENDED_VOICES
    
    @staticmethod
    def get_voice_info(voice_name: str) -> Optional[EdgeVoice]:
        """Get info for a specific voice."""
        for voice in RECOMMENDED_VOICES:
            if voice.name == voice_name:
                return voice
        return None


def get_edge_tts_client() -> EdgeTTSClient:
    """Get Edge TTS client instance."""
    return EdgeTTSClient()
