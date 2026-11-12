"""Audio processing utilities."""
import wave
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AudioData:
    """Audio data container."""
    samples: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int
    channels: int = 1


def load_wav(path: Path) -> AudioData:
    """Load WAV file as float32 numpy array."""
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        
        raw = wf.readframes(n_frames)
    
    # Convert to float32
    if sampwidth == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sampwidth == 3:
        # 24-bit - need special handling
        raw = np.frombuffer(raw, dtype=np.uint8)
        raw = raw.reshape(-1, 3)
        # Convert 24-bit to 32-bit
        samples = (raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)).astype(np.int32)
        # Sign extend
        samples = np.where(samples & 0x800000, samples | 0xFF000000, samples)
        max_val = 8388608.0
        return AudioData(samples.astype(np.float32) / max_val, sample_rate, n_channels)
    elif sampwidth == 4:
        dtype = np.int32
        max_val = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val
    
    # Handle multi-channel
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    
    return AudioData(samples, sample_rate, 1)


def save_wav(path: Path, audio: AudioData) -> None:
    """Save float32 audio as WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to int16
    samples = np.clip(audio.samples * 32767, -32768, 32767).astype(np.int16)
    
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(audio.channels)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(samples.tobytes())


def resample(audio: AudioData, target_rate: int) -> AudioData:
    """Resample audio to target sample rate."""
    if audio.sample_rate == target_rate:
        return audio
    
    # Simple linear interpolation resampling
    ratio = target_rate / audio.sample_rate
    new_length = int(len(audio.samples) * ratio)
    
    old_indices = np.arange(len(audio.samples))
    new_indices = np.arange(new_length) / ratio
    
    resampled = np.interp(new_indices, old_indices, audio.samples)
    
    return AudioData(resampled.astype(np.float32), target_rate, audio.channels)


def normalize_loudness(audio: AudioData, target_lufs: float = -16.0) -> AudioData:
    """Normalize audio loudness to target LUFS (simplified)."""
    # Measure RMS
    rms = np.sqrt(np.mean(audio.samples ** 2))
    if rms == 0:
        return audio
    
    # Convert target LUFS to RMS (approximate)
    # -16 LUFS ≈ -16 dBFS RMS for typical content
    target_rms = 10 ** (target_lufs / 20)
    
    gain = target_rms / rms
    # Limit gain to prevent clipping
    max_gain = 0.99 / np.max(np.abs(audio.samples))
    gain = min(gain, max_gain)
    
    return AudioData(audio.samples * gain, audio.sample_rate, audio.channels)


def detect_silence(
    audio: AudioData,
    threshold_db: float = -50,
    min_duration: float = 0.5
) -> list:
    """Detect silence regions in audio."""
    threshold = 10 ** (threshold_db / 20)
    
    # Compute envelope
    window = int(0.01 * audio.sample_rate)  # 10ms window
    envelope = np.abs(audio.samples)
    envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
    
    # Find silence regions
    is_silent = envelope < threshold
    silence_regions = []
    
    in_silence = False
    start = 0
    
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            start = i
        elif not silent and in_silence:
            in_silence = False
            end = i
            duration = (end - start) / audio.sample_rate
            if duration >= min_duration:
                silence_regions.append((start / audio.sample_rate, end / audio.sample_rate))
    
    # Handle silence at end
    if in_silence:
        duration = (len(is_silent) - start) / audio.sample_rate
        if duration >= min_duration:
            silence_regions.append((start / audio.sample_rate, len(is_silent) / audio.sample_rate))
    
    return silence_regions


def measure_peak(audio: AudioData) -> float:
    """Measure peak level in dBFS."""
    peak = np.max(np.abs(audio.samples))
    if peak == 0:
        return -float('inf')
    return 20 * np.log10(peak)


def measure_rms(audio: AudioData) -> float:
    """Measure RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio.samples ** 2))
    if rms == 0:
        return -float('inf')
    return 20 * np.log10(rms)


def concatenate_audio(audio_list: list) -> AudioData:
    """Concatenate multiple AudioData objects."""
    if not audio_list:
        raise ValueError("Empty audio list")
    
    # Ensure same sample rate
    target_rate = audio_list[0].sample_rate
    for a in audio_list:
        if a.sample_rate != target_rate:
            raise ValueError("All audio must have same sample rate")
    
    samples = np.concatenate([a.samples for a in audio_list])
    return AudioData(samples, target_rate)


def add_padding(audio: AudioData, pad_start: float = 0, pad_end: float = 0) -> AudioData:
    """Add silence padding to audio."""
    pad_start_samples = int(pad_start * audio.sample_rate)
    pad_end_samples = int(pad_end * audio.sample_rate)
    
    padding_start = np.zeros(pad_start_samples, dtype=np.float32)
    padding_end = np.zeros(pad_end_samples, dtype=np.float32)
    
    samples = np.concatenate([padding_start, audio.samples, padding_end])
    return AudioData(samples, audio.sample_rate, audio.channels)
