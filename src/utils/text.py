"""Text processing utilities."""
import re
import difflib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Sentence:
    """Sentence with metadata."""
    text: str
    word_start: int
    word_end: int
    is_paragraph_end: bool = False


def slugify(text: str, max_len: int = 60) -> str:
    """Convert text to URL-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug[:max_len]


def chunk_text(text: str, max_chars: int = 2500) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    chunks = []
    current = ""
    
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        if current and len(current) + len(paragraph) > max_chars:
            chunks.append(current.strip())
            current = ""
        
        current += paragraph + "\n\n"
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


def split_sentences(text: str) -> List[Sentence]:
    """Split text into sentences with word indices."""
    sentences = []
    word_index = 0
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    for para_idx, paragraph in enumerate(paragraphs):
        # Split on sentence boundaries
        sent_texts = re.split(r'(?<=[.!?])\s+', paragraph)
        
        for sent_idx, sent_text in enumerate(sent_texts):
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            
            words = sent_text.split()
            word_count = len(words)
            
            sentences.append(Sentence(
                text=sent_text,
                word_start=word_index,
                word_end=word_index + word_count,
                is_paragraph_end=(sent_idx == len(sent_texts) - 1)
            ))
            
            word_index += word_count
    
    return sentences


def estimate_duration(word_count: int, wpm: int = 140) -> float:
    """Estimate narration duration in minutes."""
    return word_count / wpm


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def similarity(a: str, b: str) -> float:
    """Calculate string similarity (0-1)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_duplicates(topics: List[str], threshold: float = 0.85) -> List[tuple]:
    """Find potential duplicate topics."""
    duplicates = []
    for i, a in enumerate(topics):
        for j, b in enumerate(topics[i+1:], i+1):
            sim = similarity(a, b)
            if sim >= threshold:
                duplicates.append((a, b, sim))
    return duplicates


def clean_script(text: str) -> str:
    """Clean script text for TTS."""
    # Remove markdown
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # code
    text = re.sub(r'#+ ', '', text)               # headers
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)    # links
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp."""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int(round((secs - int(secs)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


def parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp to seconds."""
    # Format: HH:MM:SS,mmm
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', time_str)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    return 0.0


def wrap_text(text: str, width: int = 44) -> str:
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + (1 if current else 0) > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if current_len > 0 else 0)
    
    if current:
        lines.append(" ".join(current))
    
    return "\n".join(lines)
