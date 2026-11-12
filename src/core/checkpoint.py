"""Checkpoint system for pipeline resume capability."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    stage: str
    topic: str
    topic_slug: str
    timestamp: str
    success: bool
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


class CheckpointManager:
    """Manages pipeline checkpoints for resume capability."""
    
    STAGE_ORDER = [
        'research',
        'script',
        'audio',
        'timestamps',
        'visual',
        'video',
        'publish'
    ]
    
    def __init__(self, base_dir: Path, topic_slug: str):
        self.base_dir = Path(base_dir)
        self.topic_slug = topic_slug
        self.checkpoint_dir = self.base_dir / topic_slug / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self._metadata: List[CheckpointMetadata] = self._load_metadata()
    
    def _load_metadata(self) -> List[CheckpointMetadata]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return [CheckpointMetadata(**item) for item in data]
            except Exception:
                return []
        return []
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump([asdict(m) for m in self._metadata], f, indent=2)
    
    def save(self, stage: str, data: Dict[str, Any], 
             success: bool = True, error: Optional[str] = None,
             duration_seconds: Optional[float] = None) -> Path:
        """Save checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}.json"
        
        # Add metadata to checkpoint
        checkpoint_data = {
            'stage': stage,
            'topic_slug': self.topic_slug,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'data': data
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Update metadata
        meta = CheckpointMetadata(
            stage=stage,
            topic=self.topic_slug,
            topic_slug=self.topic_slug,
            timestamp=datetime.now().isoformat(),
            success=success,
            error=error,
            duration_seconds=duration_seconds
        )
        self._metadata.append(meta)
        self._save_metadata()
        
        return checkpoint_file
    
    def load(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}.json"
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get('data')
        except Exception:
            return None
    
    def exists(self, stage: str) -> bool:
        """Check if checkpoint exists for a stage."""
        return (self.checkpoint_dir / f"{stage}.json").exists()
    
    def get_last_successful_stage(self) -> Optional[str]:
        """Get the last successfully completed stage."""
        for stage in reversed(self.STAGE_ORDER):
            if self.exists(stage):
                data = self.load(stage)
                if data is not None:
                    return stage
        return None
    
    def get_completed_stages(self) -> List[str]:
        """Get list of successfully completed stages in order."""
        completed = []
        for stage in self.STAGE_ORDER:
            if self.exists(stage):
                data = self.load(stage)
                if data is not None:
                    completed.append(stage)
        return completed
    
    def can_resume_from(self, stage: str) -> bool:
        """Check if we can resume from a specific stage."""
        try:
            idx = self.STAGE_ORDER.index(stage)
            # All previous stages must be complete
            for prev_stage in self.STAGE_ORDER[:idx]:
                if not self.exists(prev_stage):
                    return False
            return True
        except ValueError:
            return False
    
    def clear_from(self, stage: str) -> None:
        """Clear checkpoints from a stage onwards (for retry)."""
        try:
            idx = self.STAGE_ORDER.index(stage)
            for stale_stage in self.STAGE_ORDER[idx:]:
                checkpoint_file = self.checkpoint_dir / f"{stale_stage}.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            # Rebuild metadata
            self._metadata = [m for m in self._metadata if m.stage not in self.STAGE_ORDER[idx:]]
            self._save_metadata()
        except ValueError:
            pass
    
    def clear_all(self) -> None:
        """Clear all checkpoints for this topic."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self._metadata = []
    
    def get_metadata(self) -> List[CheckpointMetadata]:
        """Get all checkpoint metadata."""
        return self._metadata.copy()


def create_checkpoint_manager(output_dir: Path, topic_slug: str) -> CheckpointManager:
    """Factory function to create a checkpoint manager."""
    return CheckpointManager(output_dir, topic_slug)
