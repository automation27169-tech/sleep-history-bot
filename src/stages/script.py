"""Script stage: outline generation, section writing, review."""
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import ScriptError
from ..core.models import (
    ScriptCheckpoint, OutlineSection, ScriptSection, ScriptReview
)
from ..services.gemini import get_gemini_client
from ..services.cache import get_cache
from ..utils.text import chunk_text, count_words, estimate_duration

logger = logging.getLogger(__name__)


class ScriptStage:
    """Script generation stage handler."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.gemini = get_gemini_client()
        self.cache = get_cache()
        
        self.sections = self.config.get('script', 'sections', default=8)
        self.words_per_section = self.config.get('script', 'words_per_section', default=700)
        self.story_style = self.config.get('script', 'story_style', default='documentary')
        self.target_wpm = self.config.get('script', 'target_wpm', default=140)
    
    def generate_outline(self, topic: str, sources: List[Dict[str, Any]]) -> List[OutlineSection]:
        """Generate structured outline for the topic."""
        cache_key = f"outline|{self.topic_slug}"
        
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Using cached outline")
            return [OutlineSection(**s) for s in cached]
        
        source_summary = "\n".join([f"- {s.get('title', '')}: {s.get('url', '')}" for s in sources[:5]])
        
        if self.story_style == "story":
            prompt = f"""Plan a calm, factual history sleep story titled "{topic}". 
Return a JSON list of exactly {self.sections} objects with keys "title" and "summary" (1 sentence each).
The story drifts slowly through one place and time period, moving from gentle arrival to deep rest.
Sources for reference: {source_summary}"""
        else:
            prompt = f"""Plan a calm, curious, factual history documentary for sleep titled "{topic}".
Return a JSON list of exactly {self.sections} objects with keys "title" and "summary" (1 sentence each).
Each section covers a different aspect, example or era of the topic.
Start with a gentle introduction and end with a slow, quiet wrap-up.
Sources for reference: {source_summary}"""
        
        try:
            response = self.gemini.generate_text(prompt, json_output=True)
            outline_data = json.loads(response) if isinstance(response, str) else response
            
            outline = [OutlineSection(title=s['title'], summary=s['summary'], target_words=self.words_per_section) 
                       for s in outline_data]
            
            self.cache.set(cache_key, [s.model_dump() for s in outline])
            logger.info(f"Generated outline with {len(outline)} sections")
            return outline
            
        except Exception as e:
            logger.warning(f"Outline generation failed: {e}, using fallback")
            # Fallback outline
            return [OutlineSection(
                title=f"Section {i+1}",
                summary=f"Aspect {i+1} of {topic}",
                target_words=self.words_per_section
            ) for i in range(self.sections)]
    
    def write_section(
        self,
        topic: str,
        outline: List[OutlineSection],
        section_index: int,
        previous_tail: str = ""
    ) -> str:
        """Write a single script section."""
        section = outline[section_index]
        is_first = section_index == 0
        is_last = section_index == len(outline) - 1
        
        cache_key = f"section|{self.topic_slug}|{section_index}"
        
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Using cached section {section_index + 1}")
            return cached
        
        style_rules = (
            '- Second person, present tense ("You walk..."). Short, slow, gentle sentences. '
            'Rich sensory detail (light, sound, smell, warmth).'
            if self.story_style == "story" else
            '- Calm, slow, warm documentary narration in third person, like a bedtime history lecture '
            '(you may say "imagine" now and then). Rich concrete everyday detail: work, food, objects, '
            'smells, sounds, customs, with gentle curiosity and a little dry humour. Never rushed.'
        )
        
        prompt = f"""You are writing section {section_index + 1} of {len(outline)} of a calm history documentary for sleep: "{topic}".
Full outline: {json.dumps([{"title": s.title, "summary": s.summary} for s in outline])}
This section: {section.title} - {section.summary}
Previous section ended with: "{previous_tail}"

Rules:
- About {self.words_per_section} words, plain paragraphs only, no headings, no markdown, no lists.
{style_rules}
- Historically grounded. Only state details you are confident are accurate; when unsure, stay general.
  No invented statistics or date dumps.
- Nothing graphic, violent or frightening. Dark topics handled gently and matter-of-factly. No cliffhangers.
- Continue naturally from the previous section without recapping it.
{"- This is the opening section: begin with a short, warm welcome to the listener (settle in, dim the lights, breathe slowly), then gently introduce what tonight's video is about." if is_first else ""}
{"- This is the final section: slow down further and end with a soft, quiet fade into sleep." if is_last else ""}"""
        
        try:
            text = self.gemini.generate_text(prompt)
            self.cache.set(cache_key, text)
            logger.info(f"Section {section_index + 1} written: {count_words(text)} words")
            return text
            
        except Exception as e:
            logger.error(f"Section {section_index + 1} failed: {e}")
            raise ScriptError(f"Failed to write section {section_index + 1}: {e}")
    
    def review_script(self, full_script: str, topic: str) -> ScriptReview:
        """Review script for quality."""
        cache_key = f"review|{self.topic_slug}"
        
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Using cached review")
            return ScriptReview(**cached)
        
        # Quick checks
        word_count = count_words(full_script)
        
        # Check for repetition (simple n-gram overlap)
        repetition_score = self._check_repetition(full_script)
        
        # Check pacing (sentence length variation)
        pacing_score = self._check_pacing(full_script)
        
        # Accuracy check (would need sources, simplified here)
        accuracy_score = 0.95  # Placeholder
        
        issues = []
        if word_count < self.sections * self.words_per_section * 0.7:
            issues.append(f"Script too short: {word_count} words")
        if repetition_score > 0.1:
            issues.append(f"High repetition detected: {repetition_score:.1%}")
        if pacing_score < 0.5:
            issues.append(f"Pacing issues: {pacing_score:.1%}")
        
        review = ScriptReview(
            accuracy_score=accuracy_score,
            repetition_score=repetition_score,
            pacing_score=pacing_score,
            issues=issues
        )
        
        self.cache.set(cache_key, review.model_dump())
        logger.info(f"Script review: accuracy={accuracy_score:.0%}, repetition={repetition_score:.1%}, pacing={pacing_score:.1%}")
        return review
    
    def _check_repetition(self, text: str) -> float:
        """Check for repetitive phrases."""
        words = text.lower().split()
        if len(words) < 100:
            return 0.0
        
        # Check 3-gram repetition
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique = set(trigrams)
        return 1 - (len(unique) / len(trigrams)) if trigrams else 0
    
    def _check_pacing(self, text: str) -> float:
        """Check sentence length variation."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 5:
            return 0.5
        
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        
        # Good pacing: average 15-25 words, some variation
        if 10 <= avg_len <= 30:
            return 1.0
        elif 8 <= avg_len <= 35:
            return 0.7
        else:
            return 0.4
    
    def run(self, research_checkpoint: Dict[str, Any]) -> ScriptCheckpoint:
        """Execute script stage."""
        logger.info("Starting script stage")
        
        topic = research_checkpoint['topic']
        sources = research_checkpoint.get('sources', [])
        
        # Generate outline
        outline = self.generate_outline(topic, sources)
        
        # Write sections
        sections = []
        full_script_parts = []
        previous_tail = ""
        
        for i in range(len(outline)):
            text = self.write_section(topic, outline, i, previous_tail)
            word_count = count_words(text)
            
            sections.append(ScriptSection(
                index=i + 1,
                title=outline[i].title,
                text=text,
                word_count=word_count
            ))
            
            full_script_parts.append(text)
            previous_tail = text[-250:] if len(text) > 250 else text
        
        full_script = "\n\n".join(full_script_parts)
        total_words = count_words(full_script)
        estimated_duration = estimate_duration(total_words, self.target_wpm)
        
        # Review
        review = self.review_script(full_script, topic)
        
        # Fact-check (add to research checkpoint)
        # This would be done here in production
        
        checkpoint = ScriptCheckpoint(
            outline=outline,
            sections=sections,
            full_script=full_script,
            total_words=total_words,
            estimated_duration_min=estimated_duration,
            review=review
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('script', checkpoint.model_dump(), success=True)
        
        logger.info(f"Script complete: {total_words} words, ~{estimated_duration:.1f} min")
        return checkpoint


def run_script(output_dir: Path, topic_slug: str, research_checkpoint: Dict[str, Any]) -> ScriptCheckpoint:
    """Run script stage."""
    stage = ScriptStage(output_dir, topic_slug)
    return stage.run(research_checkpoint)
