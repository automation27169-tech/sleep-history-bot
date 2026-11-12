"""Pydantic data models for pipeline checkpoints."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class StoryStyle(str, Enum):
    DOCUMENTARY = "documentary"
    STORY = "story"


class TTSProvider(str, Enum):
    GEMINI = "gemini"
    EDGE = "edge"


class ImageProvider(str, Enum):
    POLLINATIONS = "pollinations"
    GEMINI = "gemini"


class AlignmentMethod(str, Enum):
    WHISPER = "whisper"
    ESTIMATED = "estimated"


class VisualStyle(str, Enum):
    CINEMATIC = "cinematic"
    MANUSCRIPT = "manuscript"
    ENGRAVING = "engraving"


# ============================================================
# RESEARCH STAGE MODELS
# ============================================================

class Source(BaseModel):
    title: str
    url: str
    credibility: float = Field(ge=0.0, le=1.0)


class FactCheck(BaseModel):
    claim: str
    verified: bool
    source_index: int
    notes: Optional[str] = None


class ResearchCheckpoint(BaseModel):
    topic: str
    topic_slug: str
    sources: List[Source] = []
    fact_checks: List[FactCheck] = []
    duplicate_check: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# SCRIPT STAGE MODELS
# ============================================================

class OutlineSection(BaseModel):
    title: str
    summary: str
    target_words: int = 700


class ScriptSection(BaseModel):
    index: int
    title: str
    text: str
    word_count: int


class ScriptReview(BaseModel):
    accuracy_score: float = Field(ge=0.0, le=1.0)
    repetition_score: float = Field(ge=0.0, le=1.0)
    pacing_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = []


class ScriptCheckpoint(BaseModel):
    outline: List[OutlineSection] = []
    sections: List[ScriptSection] = []
    full_script: str = ""
    total_words: int = 0
    estimated_duration_min: float = 0.0
    review: Optional[ScriptReview] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# AUDIO STAGE MODELS
# ============================================================

class AudioChunk(BaseModel):
    index: int
    path: str
    text: str
    duration: float


class AudioCheckpoint(BaseModel):
    narration_path: str
    duration_seconds: float
    chunks: List[AudioChunk] = []
    loudnorm_applied: bool = True
    sample_rate: int = 24000
    channels: int = 1
    tts_provider: TTSProvider = TTSProvider.GEMINI
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# TIMESTAMPS STAGE MODELS
# ============================================================

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class SentenceTimestamp(BaseModel):
    text: str
    start: float
    end: float
    word_start_idx: int
    word_end_idx: int


class TimestampsCheckpoint(BaseModel):
    word_timestamps: List[WordTimestamp] = []
    sentences: List[SentenceTimestamp] = []
    srt_path: str = ""
    alignment_method: AlignmentMethod = AlignmentMethod.WHISPER
    alignment_quality: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# VISUAL STAGE MODELS
# ============================================================

class Scene(BaseModel):
    index: int
    start: float
    end: float
    text: str
    prompt: str
    style: VisualStyle = VisualStyle.CINEMATIC
    image_path: str = ""
    is_broll: bool = False


class ThumbnailVariant(BaseModel):
    variant: int
    path: str


class VisualCheckpoint(BaseModel):
    scenes: List[Scene] = []
    thumbnails: List[ThumbnailVariant] = []
    image_provider: ImageProvider = ImageProvider.POLLINATIONS
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# VIDEO STAGE MODELS
# ============================================================

class VideoCheckpoint(BaseModel):
    video_path: str
    duration_seconds: float
    resolution: List[int] = [1920, 1080]
    fps: int = 24
    codec: str = "libx264"
    crf: int = 23
    bitrate_kbps: int = 3500
    audio_codec: str = "aac"
    audio_bitrate: int = 160
    has_subtitles: bool = False
    has_ambience: bool = False
    render_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# PUBLISH STAGE MODELS
# ============================================================

class QualityCheck(BaseModel):
    name: str
    target: str
    actual: str
    pass_: bool = Field(alias="pass")


class QualityReport(BaseModel):
    duration_check: QualityCheck
    audio_check: QualityCheck
    video_check: QualityCheck
    content_check: QualityCheck
    overall_pass: bool


class Metadata(BaseModel):
    title: str
    description: str
    tags: List[str] = []
    chapters: List[str] = []


class PublishCheckpoint(BaseModel):
    video_path: str
    thumbnails: List[str] = []
    metadata: Metadata
    quality_report: QualityReport
    approval_ready: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# CHECKPOINT WRAPPER
# ============================================================

class StageCheckpoint(BaseModel):
    stage: str
    topic_slug: str
    timestamp: datetime
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
