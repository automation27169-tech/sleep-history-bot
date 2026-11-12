"""Google Gemini API wrapper with retry logic and model fallback."""
import os
import time
import random
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path

from google import genai
from google.genai import types

from ..core.config import get_config
from ..core.exceptions import GeminiError, RetryExhaustedError
from ..services.cache import get_cache

logger = logging.getLogger(__name__)

# Transient error patterns that warrant retry
TRANSIENT_ERRORS = (
    "503", "429", "500", "502", "504", "UNAVAILABLE", "overloaded",
    "RESOURCE_EXHAUSTED", "timed out", "Connection", "DEADLINE_EXCEEDED",
    "internal error", "service unavailable"
)


class GeminiClient:
    """Wrapper for Google Gemini API with model fallback and caching."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY not provided")
        
        self.client = genai.Client(api_key=self.api_key)
        self.config = get_config()
        self.cache = get_cache()
        
        # Model lists from config with fallbacks
        self.text_models = self.config.get('gemini', 'text_models', default=[
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ])
        self.tts_models = self.config.get('gemini', 'tts_models', default=[
            "gemini-2.5-flash-preview-tts",
            "gemini-2.0-flash-preview-tts"
        ])
        self.image_models = self.config.get('gemini', 'image_models', default=[
            "gemini-2.0-flash-preview-image-generation"
        ])
        
        self.max_tokens = self.config.get('gemini', 'max_tokens', default=8192)
        self.temperature = self.config.get('gemini', 'temperature', default=0.7)
        self.top_p = self.config.get('gemini', 'top_p', default=0.9)
        
        # Retry settings
        self.max_retries = self.config.get('pipeline', 'retry_attempts', default=3)
        self.base_delay = self.config.get('pipeline', 'retry_base_delay', default=15)
        self.max_delay = self.config.get('pipeline', 'max_retry_delay', default=180)
    
    def _is_transient(self, error: Exception) -> bool:
        """Check if error is transient and worth retrying."""
        msg = str(error)
        return any(t in msg for t in TRANSIENT_ERRORS)
    
    def _retry_with_backoff(self, fn, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1 or not self._is_transient(e):
                    raise
                
                wait = min(self.base_delay * (2 ** attempt), self.max_delay)
                wait += random.randint(0, 10)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        
        raise RetryExhaustedError("gemini_request", self.max_retries, last_error)
    
    def _try_models(self, models: List[str], call_fn, *args, **kwargs):
        """Try multiple models in order until one succeeds."""
        last_error = None
        
        for model in models:
            try:
                logger.debug(f"Trying model: {model}")
                return call_fn(model, *args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model} failed: {e}")
                if not self._is_transient(e):
                    # Non-transient error, try next model
                    continue
                # Transient error - retry this model
                try:
                    return self._retry_with_backoff(lambda m=model: call_fn(m, *args, **kwargs))
                except RetryExhaustedError:
                    # All retries exhausted for this model, try next
                    continue
        
        raise GeminiError(f"All models failed. Last error: {last_error}")
    
    # ============================================================
    # TEXT GENERATION
    # ============================================================
    
    def generate_text(
        self,
        prompt: str,
        json_output: bool = False,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """Generate text using Gemini with model fallback."""
        cache_key = f"gemini_text|{hashlib.sha256(prompt.encode()).hexdigest()[:16]}|{json_output}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for text generation")
                return cached
        
        def call_model(model: str) -> str:
            contents = prompt
            if system_prompt:
                contents = f"{system_prompt}\n\n{prompt}"
            
            cfg = None
            if json_output:
                cfg = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            else:
                cfg = types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg
            )
            
            if response.text:
                return response.text.strip()
            raise GeminiError("Empty response from model")
        
        result = self._try_models(self.text_models, call_model)
        
        if use_cache:
            self.cache.set(cache_key, result)
        
        return result
    
    # ============================================================
    # TTS GENERATION
    # ============================================================
    
    def generate_tts(
        self,
        text: str,
        voice: str = "Algieba",
        narration_style: str = "",
        use_cache: bool = True
    ) -> bytes:
        """Generate TTS audio using Gemini."""
        full_text = f"{narration_style}{text}" if narration_style else text
        cache_key = f"gemini_tts|{voice}|{hashlib.sha256(full_text.encode()).hexdigest()[:16]}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for TTS")
                return bytes(cached)  # Convert back to bytes
        
        def call_model(model: str) -> bytes:
            response = self.client.models.generate_content(
                model=model,
                contents=full_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    )
                )
            )
            
            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
            raise GeminiError("No audio data in response")
        
        audio_data = self._try_models(self.tts_models, call_model)
        
        if use_cache:
            # Cache as list of ints for JSON serialization
            self.cache.set(cache_key, list(audio_data))
        
        return audio_data
    
    # ============================================================
    # IMAGE GENERATION
    # ============================================================
    
    def generate_image(
        self,
        prompt: str,
        use_cache: bool = True
    ) -> bytes:
        """Generate image using Gemini (if available)."""
        cache_key = f"gemini_image|{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for image generation")
                return bytes(cached)
        
        def call_model(model: str) -> bytes:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )
            
            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
            raise GeminiError("No image data in response")
        
        image_data = self._try_models(self.image_models, call_model)
        
        if use_cache:
            self.cache.set(cache_key, list(image_data))
        
        return image_data


import hashlib

# Convenience function for backward compatibility
def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client instance."""
    return GeminiClient()
