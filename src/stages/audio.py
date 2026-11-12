"""Audio stage: TTS generation, concatenation, loudness normalization."""
import logging
import wave
import tempfile
from pathlib import Path
from typing import List, Optional

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import AudioError
from ..core.models import AudioCheckpoint, AudioChunk
from ..services.gemini import get_gemini_client
from ..services.edge_tts import get_edge_tts_client
from ..services.cache import get_cache
from ..utils.text import chunk_text
from ..utils.audio import load_wav, save_wav, concatenate_audio, add_padding

logger = logging.getLogger(__name__)


class AudioStage:
    """Audio generation stage handler."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.gemini = get_gemini_client()
        self.edge_tts = get_edge_tts_client()
        self.cache = get_cache()
        
        self.chunk_size = self.config.get('audio', 'chunk_size_chars', default=2500)
        self.gemini_voice = self.config.get('audio', 'gemini_voice', default='Algieba')
        self.edge_voice = self.config.get('audio', 'edge_voice', default='en-US-GuyNeural')
        self.narration_style = self.config.get('audio', 'narration_style', 
            default='very slow, soft, warm, calm and soothing, with long gentle pauses between sentences')
        self.target_loudness = self.config.get('audio', 'target_loudness', default=-16)
        self.target_lra = self.config.get('audio', 'target_lra', default=7)
        self.target_tp = self.config.get('audio', 'target_tp', default=-1.5)
        self.padding_end = self.config.get('audio', 'padding_end_seconds', default=6)
        self.sample_rate = self.config.get('audio', 'sample_rate', default=24000)
    
    def synthesize_chunks_gemini(self, chunks: List[str]) -> List[bytes]:
        """Synthesize chunks using Gemini TTS."""
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            cache_key = f"tts_gemini|{self.gemini_voice}|{hash(chunk)}"
            
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for Gemini TTS chunk {i+1}")
                audio_chunks.append(bytes(cached))
                continue
            
            logger.info(f"Gemini TTS chunk {i+1}/{len(chunks)}")
            audio_data = self.gemini.generate_tts(
                chunk,
                voice=self.gemini_voice,
                narration_style=self.narration_style
            )
            audio_chunks.append(audio_data)
            self.cache.set(cache_key, list(audio_data))
        
        return audio_chunks
    
    def synthesize_chunks_edge(self, chunks: List[str]) -> List[bytes]:
        """Synthesize chunks using Edge TTS."""
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            cache_key = f"tts_edge|{self.edge_voice}|{hash(chunk)}"
            
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for Edge TTS chunk {i+1}")
                audio_chunks.append(bytes(cached))
                continue
            
            logger.info(f"Edge TTS chunk {i+1}/{len(chunks)}")
            audio_data = self.edge_tts.synthesize(
                chunk,
                voice=self.edge_voice,
                rate="-15%",
                pitch="-5Hz",
                sample_rate=self.sample_rate
            )
            audio_chunks.append(audio_data)
            self.cache.set(cache_key, list(audio_data))
        
        return audio_chunks
    
    def save_chunks_as_wav(self, audio_chunks: List[bytes], output_dir: Path) -> List[Path]:
        """Save audio chunks as WAV files."""
        chunk_paths = []
        
        for i, audio_data in enumerate(audio_chunks):
            chunk_path = output_dir / f"voice_{i:03d}.wav"
            
            # Parse WAV header to get format
            with wave.open(str(chunk_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    def concatenate_and_normalize(
        self,
        chunk_paths: List[Path],
        output_path: Path
    ) -> float:
        """Concatenate chunks and apply loudness normalization."""
        # Load all chunks
        audio_list = [load_wav(p) for p in chunk_paths]
        
        # Concatenate
        combined = concatenate_audio(audio_list)
        
        # Add padding at end
        combined = add_padding(combined, pad_end=self.padding_end)
        
        # Normalize loudness (simplified - would use FFmpeg loudnorm in production)
        # For now just save
        save_wav(output_path, combined)
        
        # Get duration
        duration = len(combined.samples) / combined.sample_rate
        
        logger.info(f"Audio concatenated: {duration:.1f}s")
        return duration
    
    def verify_audio(self, audio_path: Path) -> dict:
        """Verify audio quality."""
        audio = load_wav(audio_path)
        
        duration = len(audio.samples) / audio.sample_rate
        peak_db = 20 * np.log10(np.max(np.abs(audio.samples)) + 1e-10)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio.samples**2)) + 1e-10)
        
        # Check for clipping
        clipping = np.sum(np.abs(audio.samples) > 0.99) / len(audio.samples) > 0.001
        
        # Check for silence gaps
        silence_threshold = 10 ** (-50 / 20)  # -50 dB
        silent = np.abs(audio.samples) < silence_threshold
        silence_gaps = 0
        in_silence = False
        for s in silent:
            if s and not in_silence:
                in_silence = True
            elif not s and in_silence:
                in_silence = False
                silence_gaps += 1
        
        return {
            'duration': duration,
            'peak_db': peak_db,
            'rms_db': rms_db,
            'clipping': clipping,
            'silence_gaps': silence_gaps,
            'sample_rate': audio.sample_rate,
            'channels': audio.channels
        }
    
    def run(self, script_checkpoint: Dict[str, Any]) -> AudioCheckpoint:
        """Execute audio stage."""
        logger.info("Starting audio stage")
        
        script = script_checkpoint['full_script']
        output_dir = self.output_dir / self.topic_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chunk script for TTS
        chunks = chunk_text(script, self.chunk_size)
        logger.info(f"Script split into {len(chunks)} chunks")
        
        # Try Gemini TTS first, fallback to Edge TTS
        try:
            audio_data = self.synthesize_chunks_gemini(chunks)
            tts_provider = "gemini"
        except Exception as e:
            logger.warning(f"Gemini TTS failed: {e}, falling back to Edge TTS")
            # Clear any partial Gemini chunks from cache
            for i in range(len(chunks)):
                cache_key = f"tts_gemini|{self.gemini_voice}|{hash(chunks[i])}"
                self.cache.delete(cache_key)
            audio_data = self.synthesize_chunks_edge(chunks)
            tts_provider = "edge"
        
        # Save chunks
        chunk_paths = self.save_chunks_as_wav(audio_data, output_dir)
        
        # Create concat list
        narration_raw = output_dir / "narration_raw.wav"
        list_path = output_dir / "voice_list.txt"
        with open(list_path, 'w') as f:
            for p in chunk_paths:
                f.write(f"file '{p.resolve()}'\n")
        
        # Concatenate using FFmpeg (more reliable)
        import subprocess
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-c", "copy", str(narration_raw)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise AudioError(f"FFmpeg concat failed: {result.stderr}")
        
        # Apply loudnorm
        narration_final = output_dir / "narration.wav"
        import subprocess
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(narration_raw),
            "-af", f"loudnorm=I={self.target_loudness}:LRA={self.target_lra}:TP={self.target_tp},apad=pad_dur={self.padding_end}",
            str(narration_final)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise AudioError(f"FFmpeg loudnorm failed: {result.stderr}")
        
        # Get duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(narration_final)
        ], capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        # Build chunk info
        chunk_infos = []
        for i, p in enumerate(chunk_paths):
            dur_result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(p)
            ], capture_output=True, text=True)
            chunk_dur = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 0
            chunk_infos.append(AudioChunk(
                index=i,
                path=str(p),
                text=chunks[i][:100] + "...",
                duration=chunk_dur
            ))
        
        # Cleanup temp files
        narration_raw.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)
        
        checkpoint = AudioCheckpoint(
            narration_path=str(narration_final),
            duration_seconds=duration,
            chunks=chunk_infos,
            loudnorm_applied=True,
            sample_rate=self.sample_rate,
            channels=1,
            tts_provider=tts_provider
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('audio', checkpoint.model_dump(), success=True)
        
        logger.info(f"Audio complete: {duration:.1f}s ({duration/60:.1f} min)")
        return checkpoint


def run_audio(output_dir: Path, topic_slug: str, script_checkpoint: Dict[str, Any]) -> AudioCheckpoint:
    """Run audio stage."""
    stage = AudioStage(output_dir, topic_slug)
    return stage.run(script_checkpoint)
