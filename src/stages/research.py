"""Research stage: topic selection, fact-checking, source gathering."""
import logging
import re
from pathlib import Path
from typing import List, Optional
from difflib import SequenceMatcher

from ..core.config import get_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.exceptions import ResearchError, DuplicateTopicError, TopicExhaustedError
from ..core.models import ResearchCheckpoint, Source, FactCheck
from ..services.gemini import get_gemini_client
from ..services.cache import get_cache
from ..utils.text import similarity, slugify

logger = logging.getLogger(__name__)


class ResearchStage:
    """Research stage handler."""
    
    def __init__(self, output_dir: Path, topic_slug: str):
        self.output_dir = output_dir
        self.topic_slug = topic_slug
        self.checkpoint_mgr = create_checkpoint_manager(output_dir, topic_slug)
        self.config = get_config()
        self.gemini = get_gemini_client()
        self.cache = get_cache()
    
    def select_topic(self, topic_override: Optional[str] = None) -> str:
        """Select next topic from queue."""
        if topic_override:
            return topic_override
        
        topics_file = Path(self.config.get('topics', 'file', default='data/topics.txt'))
        done_file = Path(self.config.get('topics', 'done_file', default='data/done.txt'))
        threshold = self.config.get('topics', 'similarity_threshold', default=0.85)
        
        # Load topics
        if not topics_file.exists():
            raise TopicExhaustedError("topics.txt not found")
        
        all_topics = [line.strip() for line in topics_file.read_text().splitlines() if line.strip()]
        
        # Load done topics
        done_topics = set()
        if done_file.exists():
            done_topics = set(line.strip() for line in done_file.read_text().splitlines() if line.strip())
        
        # Find next undone topic
        for topic in all_topics:
            if topic not in done_topics:
                # Check for fuzzy duplicates
                is_duplicate = False
                for done in done_topics:
                    if similarity(topic, done) >= threshold:
                        is_duplicate = True
                        logger.warning(f"Topic similar to done: '{topic}' ~ '{done}'")
                        break
                
                if not is_duplicate:
                    return topic
        
        raise TopicExhaustedError("All topics completed")
    
    def verify_unique(self, topic: str) -> bool:
        """Verify topic hasn't been covered."""
        done_file = Path(self.config.get('topics', 'done_file', default='data/done.txt'))
        threshold = self.config.get('topics', 'similarity_threshold', default=0.85)
        
        if not done_file.exists():
            return True
        
        done_topics = [line.strip() for line in done_file.read_text().splitlines() if line.strip()]
        
        for done in done_topics:
            if similarity(topic, done) >= threshold:
                return False
        
        return True
    
    def gather_sources(self, topic: str) -> List[Source]:
        """Gather historical sources for topic using Gemini."""
        cache_key = f"sources|{slugify(topic)}"
        
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Using cached sources")
            return [Source(**s) for s in cached]
        
        prompt = f"""Find credible historical sources for a documentary about: "{topic}"

Return JSON list of sources with:
- title: source title
- url: URL or citation
- credibility: 0.0-1.0 (academic=0.9+, museum=0.8, reputable site=0.7, blog=0.5)

Focus on: academic papers, museum collections, historical societies, primary sources, 
reputable history websites (Britannica, History.com, Smithsonian, etc.).
Minimum 5 sources, maximum 10."""
        
        try:
            response = self.gemini.generate_text(prompt, json_output=True)
            sources_data = eval(response) if isinstance(response, str) else response
            
            sources = [Source(**s) for s in sources_data]
            
            # Cache
            self.cache.set(cache_key, [s.model_dump() for s in sources])
            
            logger.info(f"Gathered {len(sources)} sources for: {topic}")
            return sources
            
        except Exception as e:
            logger.warning(f"Source gathering failed: {e}, using empty list")
            return []
    
    def fact_check_script(self, script: str, sources: List[Source]) -> List[FactCheck]:
        """Fact-check claims in script against sources."""
        if not sources:
            return []
        
        cache_key = f"factcheck|{slugify(script[:100])}"
        
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Using cached fact-checks")
            return [FactCheck(**f) for f in cached]
        
        # Extract verifiable claims (dates, names, numbers, specific events)
        claims = self._extract_claims(script)
        
        if not claims:
            return []
        
        source_text = "\n".join([f"[{i}] {s.title}: {s.url}" for i, s in enumerate(sources)])
        
        prompt = f"""Fact-check these historical claims against the provided sources.
Return JSON list with: claim, verified (true/false), source_index (which source), notes.

Sources:
{source_text}

Claims to verify:
{chr(10).join(f"- {c}" for c in claims[:20])}"""  # Limit to 20 claims
        
        try:
            response = self.gemini.generate_text(prompt, json_output=True)
            fact_checks_data = eval(response) if isinstance(response, str) else response
            
            fact_checks = [FactCheck(**f) for f in fact_checks_data]
            
            self.cache.set(cache_key, [f.model_dump() for f in fact_checks])
            
            logger.info(f"Fact-checked {len(fact_checks)} claims")
            return fact_checks
            
        except Exception as e:
            logger.warning(f"Fact-checking failed: {e}")
            return []
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        claims = []
        
        # Patterns for verifiable claims
        patterns = [
            r'\b(in|during|around)\s+\d{1,4}(?:\s*(?:BC|AD|BCE|CE))?\b',  # Dates
            r'\b\d{1,3}(?:,\d{3})*\s+(?:people|workers|soldiers|citizens|inhabitants)\b',  # Numbers
            r'\b(?:King|Queen|Emperor|Pharaoh|President|General|Pope)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Titles + names
            r'\b(?:Battle of|Treaty of|War of|Siege of)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Events
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        
        # Also extract sentences with specific details
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if any(word in sent.lower() for word in ['date', 'year', 'born', 'died', 'ruled', 'built', 'founded']):
                if len(sent) > 30 and len(sent) < 200:
                    claims.append(sent.strip())
        
        return list(set(claims))[:30]  # Deduplicate and limit
    
    def run(self, topic_override: Optional[str] = None) -> ResearchCheckpoint:
        """Execute research stage."""
        logger.info("Starting research stage")
        
        # Select topic
        topic = self.select_topic(topic_override)
        topic_slug = slugify(topic)
        
        # Verify uniqueness
        is_unique = self.verify_unique(topic)
        if not is_unique:
            raise DuplicateTopicError(topic)
        
        # Gather sources
        sources = self.gather_sources(topic)
        
        # Create checkpoint (fact-checks will be done after script generation)
        checkpoint = ResearchCheckpoint(
            topic=topic,
            topic_slug=topic_slug,
            sources=sources,
            fact_checks=[],
            duplicate_check=is_unique
        )
        
        # Save checkpoint
        self.checkpoint_mgr.save('research', checkpoint.model_dump(), success=True)
        
        logger.info(f"Research complete: {topic}")
        return checkpoint


def run_research(output_dir: Path, topic_slug: str, topic_override: Optional[str] = None) -> ResearchCheckpoint:
    """Run research stage."""
    stage = ResearchStage(output_dir, topic_slug)
    return stage.run(topic_override)
