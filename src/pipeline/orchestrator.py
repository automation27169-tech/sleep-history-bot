"""Main pipeline orchestrator."""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core.config import get_config, reload_config
from ..core.checkpoint import create_checkpoint_manager
from ..core.logging import setup_logging
from ..core.exceptions import SleepHistoryError, StageError

# Import stage runners
from ..stages.research import run_research
from ..stages.script import run_script
from ..stages.audio import run_audio
from ..stages.timestamps import run_timestamps
from ..stages.visual import run_visual
from ..stages.video import run_video
from ..stages.publish import run_publish

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    'research',
    'script', 
    'audio',
    'timestamps',
    'visual',
    'video',
    'publish'
]

STAGE_RUNNERS = {
    'research': run_research,
    'script': run_script,
    'audio': run_audio,
    'timestamps': run_timestamps,
    'visual': run_visual,
    'video': run_video,
    'publish': run_publish,
}

STAGE_DEPS = {
    'research': [],
    'script': ['research'],
    'audio': ['script'],
    'timestamps': ['script', 'audio'],
    'visual': ['timestamps', 'script'],
    'video': ['visual', 'audio', 'timestamps'],
    'publish': ['video', 'audio', 'timestamps', 'visual', 'script', 'research'],
}


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    topic: str
    topic_slug: str
    completed_stages: List[str]
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    checkpoints: Dict[str, Any] = None
    duration_seconds: float = 0.0


class PipelineOrchestrator:
    """Orchestrates the full video production pipeline."""
    
    def __init__(self, output_dir: Path, config_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        if config_path:
            reload_config(config_path)
        self.config = get_config()
        
        # Setup logging
        self.logger = setup_logging()
        
        self.checkpoints: Dict[str, Any] = {}
    
    def run_stage(
        self,
        stage: str,
        topic_slug: str,
        topic_override: Optional[str] = None,
        force: bool = False
    ) -> Any:
        """Run a single pipeline stage."""
        self.logger.info(f"=== Running stage: {stage} ===")
        start_time = time.time()
        
        # Get checkpoint manager
        checkpoint_mgr = create_checkpoint_manager(self.output_dir, topic_slug)
        
        # Check if already completed
        if not force and checkpoint_mgr.exists(stage):
            self.logger.info(f"Stage {stage} already completed, loading checkpoint")
            return checkpoint_mgr.load(stage)
        
        # Check dependencies
        for dep in STAGE_DEPS.get(stage, []):
            if not checkpoint_mgr.exists(dep):
                raise StageError(stage, f"Dependency '{dep}' not completed")
        
        # Load dependency checkpoints
        deps = {}
        for dep in STAGE_DEPS.get(stage, []):
            deps[dep] = checkpoint_mgr.load(dep)
            if deps[dep] is None:
                raise StageError(stage, f"Dependency '{dep}' checkpoint not found")
        
        # Run stage
        runner = STAGE_RUNNERS[stage]
        
        try:
            if stage == 'research':
                result = runner(self.output_dir, topic_slug, topic_override)
            elif stage == 'script':
                result = runner(self.output_dir, topic_slug, deps['research'])
            elif stage == 'audio':
                result = runner(self.output_dir, topic_slug, deps['script'])
            elif stage == 'timestamps':
                result = runner(self.output_dir, topic_slug, deps['script'], deps['audio'])
            elif stage == 'visual':
                result = runner(self.output_dir, topic_slug, deps['timestamps'], deps['script'])
            elif stage == 'video':
                result = runner(self.output_dir, topic_slug, deps['visual'], deps['audio'], deps['timestamps'])
            elif stage == 'publish':
                result = runner(self.output_dir, topic_slug, deps['video'], deps['audio'], 
                               deps['timestamps'], deps['visual'], deps['script'], deps['research'])
            else:
                raise StageError(stage, f"Unknown stage: {stage}")
            
            self.checkpoints[stage] = result.model_dump() if hasattr(result, 'model_dump') else result
            
            elapsed = time.time() - start_time
            self.logger.info(f"Stage {stage} completed in {elapsed:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Stage {stage} failed: {e}")
            # Save failed checkpoint
            checkpoint_mgr.save(stage, {}, success=False, error=str(e))
            raise
    
    def run_full_pipeline(
        self,
        topic_override: Optional[str] = None,
        start_from: Optional[str] = None,
        stages: Optional[List[str]] = None
    ) -> PipelineResult:
        """Run the complete pipeline."""
        start_time = time.time()
        
        # Determine stages to run
        stages_to_run = stages or STAGE_ORDER
        
        # Handle resume
        if start_from:
            try:
                idx = STAGE_ORDER.index(start_from)
                stages_to_run = STAGE_ORDER[idx:]
            except ValueError:
                raise ValueError(f"Unknown stage: {start_from}")
        
        # Run first stage (research) to get topic
        if 'research' in stages_to_run:
            research_result = self.run_stage('research', 'temp', topic_override)
            topic = research_result.topic
            topic_slug = research_result.topic_slug
            
            # Recreate checkpoint manager with correct topic_slug
            self.checkpoints['research'] = research_result.model_dump()
        else:
            # Need to determine topic from existing checkpoint
            # This would need the topic_slug to be known
            raise ValueError("Must start from research stage or provide topic_slug")
        
        # Run remaining stages
        completed = ['research']
        
        for stage in stages_to_run[1:]:
            try:
                result = self.run_stage(stage, topic_slug)
                self.checkpoints[stage] = result.model_dump() if hasattr(result, 'model_dump') else result
                completed.append(stage)
            except Exception as e:
                return PipelineResult(
                    success=False,
                    topic=topic,
                    topic_slug=topic_slug,
                    completed_stages=completed,
                    failed_stage=stage,
                    error=str(e),
                    checkpoints=self.checkpoints,
                    duration_seconds=time.time() - start_time
                )
        
        return PipelineResult(
            success=True,
            topic=topic,
            topic_slug=topic_slug,
            completed_stages=completed,
            checkpoints=self.checkpoints,
            duration_seconds=time.time() - start_time
        )
    
    def run_single_stage(
        self,
        stage: str,
        topic_slug: str,
        topic_override: Optional[str] = None,
        force: bool = False
    ) -> Any:
        """Run a single stage by name."""
        return self.run_stage(stage, topic_slug, topic_override, force)
    
    def get_status(self, topic_slug: str) -> Dict[str, Any]:
        """Get pipeline status for a topic."""
        checkpoint_mgr = create_checkpoint_manager(self.output_dir, topic_slug)
        
        status = {
            'topic_slug': topic_slug,
            'stages': {}
        }
        
        for stage in STAGE_ORDER:
            exists = checkpoint_mgr.exists(stage)
            if exists:
                data = checkpoint_mgr.load(stage)
                status['stages'][stage] = {
                    'completed': True,
                    'has_data': data is not None
                }
            else:
                status['stages'][stage] = {'completed': False}
        
        return status


def create_orchestrator(output_dir: Path, config_path: Optional[str] = None) -> PipelineOrchestrator:
    """Factory function to create pipeline orchestrator."""
    return PipelineOrchestrator(output_dir, config_path)
