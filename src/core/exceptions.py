"""Custom exceptions for Sleep History Bot."""


class SleepHistoryError(Exception):
    """Base exception for Sleep History Bot."""
    pass


class ConfigurationError(SleepHistoryError):
    """Configuration loading or validation error."""
    pass


class CheckpointError(SleepHistoryError):
    """Checkpoint save/load error."""
    pass


class StageError(SleepHistoryError):
    """Pipeline stage execution error."""
    def __init__(self, stage: str, message: str, original_error: Exception = None):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"Stage '{stage}' failed: {message}")


class ResearchError(StageError):
    """Research stage error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("research", message, original_error)


class ScriptError(StageError):
    """Script generation error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("script", message, original_error)


class AudioError(StageError):
    """Audio generation error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("audio", message, original_error)


class TimestampsError(StageError):
    """Timestamp alignment error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("timestamps", message, original_error)


class VisualError(StageError):
    """Visual generation error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("visual", message, original_error)


class VideoError(StageError):
    """Video rendering error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("video", message, original_error)


class PublishError(StageError):
    """Publish package error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("publish", message, original_error)


class QualityGateError(SleepHistoryError):
    """Quality gate validation failure."""
    def __init__(self, gate: str, message: str, details: dict = None):
        self.gate = gate
        self.details = details or {}
        super().__init__(f"Quality gate '{gate}' failed: {message}")


class ServiceError(SleepHistoryError):
    """External service error."""
    def __init__(self, service: str, message: str, original_error: Exception = None):
        self.service = service
        self.original_error = original_error
        super().__init__(f"Service '{service}' error: {message}")


class GeminiError(ServiceError):
    """Gemini API error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("gemini", message, original_error)


class PollinationsError(ServiceError):
    """Pollinations API error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("pollinations", message, original_error)


class EdgeTTSError(ServiceError):
    """Edge-TTS error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("edge_tts", message, original_error)


class WhisperError(ServiceError):
    """Whisper alignment error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("whisper", message, original_error)


class FFmpegError(ServiceError):
    """FFmpeg processing error."""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("ffmpeg", message, original_error)


class TopicExhaustedError(SleepHistoryError):
    """No more topics available."""
    pass


class DuplicateTopicError(SleepHistoryError):
    """Topic already covered."""
    def __init__(self, topic: str):
        self.topic = topic
        super().__init__(f"Topic already covered: {topic}")


class ValidationError(SleepHistoryError):
    """Data validation error."""
    pass


class RetryExhaustedError(SleepHistoryError):
    """All retry attempts exhausted."""
    def __init__(self, operation: str, attempts: int, last_error: Exception = None):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Operation '{operation}' failed after {attempts} attempts")
