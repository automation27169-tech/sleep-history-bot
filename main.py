#!/usr/bin/env python3
"""Sleep History Bot - Main CLI Entry Point.

Zero-cost automated History for Sleep YouTube channel production.
"""
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import get_config, reload_config
from src.core.logging import setup_logging
from src.core.exceptions import SleepHistoryError
from src.pipeline.orchestrator import create_orchestrator, PipelineOrchestrator


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def cmd_full(args):
    """Run full pipeline."""
    config_path = args.config
    if config_path:
        reload_config(config_path)
    
    orchestrator = create_orchestrator(Path(args.output), config_path)
    
    result = orchestrator.run_full_pipeline(
        topic_override=args.topic,
        start_from=args.resume_from,
        stages=args.stages.split(',') if args.stages else None
    )
    
    if result.success:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   Topic: {result.topic}")
        print(f"   Duration: {result.duration_seconds:.1f}s")
        print(f"   Stages: {', '.join(result.completed_stages)}")
        print(f"   Output: {args.output}/{result.topic_slug}/")
        return 0
    else:
        print(f"\n❌ Pipeline failed at stage: {result.failed_stage}")
        print(f"   Error: {result.error}")
        print(f"   Completed: {', '.join(result.completed_stages)}")
        return 1


def cmd_stage(args):
    """Run a single stage."""
    config_path = args.config
    if config_path:
        reload_config(config_path)
    
    orchestrator = create_orchestrator(Path(args.output), config_path)
    
    try:
        result = orchestrator.run_single_stage(
            args.stage,
            args.topic_slug,
            topic_override=args.topic,
            force=args.force
        )
        print(f"\n✅ Stage '{args.stage}' completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Stage '{args.stage}' failed: {e}")
        return 1


def cmd_status(args):
    """Show pipeline status."""
    config_path = args.config
    if config_path:
        reload_config(config_path)
    
    orchestrator = create_orchestrator(Path(args.output), config_path)
    status = orchestrator.get_status(args.topic_slug)
    
    print(f"\nPipeline Status: {status['topic_slug']}")
    print("-" * 50)
    for stage, info in status['stages'].items():
        marker = "✅" if info['completed'] else "⏳"
        data = " (data)" if info.get('has_data') else ""
        print(f"  {marker} {stage}{data}")
    return 0


def cmd_topic(args):
    """Topic management commands."""
    config = get_config()
    topics_file = Path(config.get('topics', 'file', default='data/topics.txt'))
    done_file = Path(config.get('topics', 'done_file', default='data/done.txt'))
    
    if args.list:
        if topics_file.exists():
            topics = [l.strip() for l in topics_file.read_text().splitlines() if l.strip()]
            done = set()
            if done_file.exists():
                done = set(l.strip() for l in done_file.read_text().splitlines() if l.strip())
            
            print("\nTopics:")
            for i, topic in enumerate(topics, 1):
                status = "✅" if topic in done else "⏳"
                print(f"  {i:2d}. {status} {topic}")
        else:
            print("No topics.txt found")
        return 0
    
    if args.next:
        if topics_file.exists():
            topics = [l.strip() for l in topics_file.read_text().splitlines() if l.strip()]
            done = set()
            if done_file.exists():
                done = set(l.strip() for l in done_file.read_text().splitlines() if l.strip())
            
            for topic in topics:
                if topic not in done:
                    print(topic)
                    return 0
            print("No topics remaining")
        return 0
    
    return 0


def cmd_clean(args):
    """Clean output/checkpoints."""
    output_dir = Path(args.output)
    topic_slug = args.topic_slug
    
    if topic_slug:
        # Clean specific topic
        topic_dir = output_dir / topic_slug
        if topic_dir.exists():
            import shutil
            shutil.rmtree(topic_dir)
            print(f"Cleaned: {topic_dir}")
        else:
            print(f"Not found: {topic_dir}")
    else:
        # Clean all
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"Cleaned all: {output_dir}")
        else:
            print(f"Not found: {output_dir}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Sleep History Bot - Automated History for Sleep Video Production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (auto-select next topic)
  python main.py full
  
  # Run full pipeline with specific topic
  python main.py full --topic "The Strangest Jobs in Ancient Rome"
  
  # Resume from video stage
  python main.py full --resume-from video
  
  # Run single stage
  python main.py stage --stage script --topic-slug the-strangest-jobs-in-ancient-rome
  
  # Check status
  python main.py status --topic-slug the-strangest-jobs-in-ancient-rome
  
  # List topics
  python main.py topic --list
  
  # Get next topic
  python main.py topic --next
        """
    )
    
    parser.add_argument('--config', help='Path to config YAML')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Full pipeline
    p_full = subparsers.add_parser('full', help='Run full pipeline')
    p_full.add_argument('--topic', help='Override topic (default: next from topics.txt)')
    p_full.add_argument('--resume-from', help='Resume from stage')
    p_full.add_argument('--stages', help='Comma-separated list of stages to run')
    p_full.set_defaults(func=cmd_full)
    
    # Single stage
    p_stage = subparsers.add_parser('stage', help='Run single stage')
    p_stage.add_argument('--stage', required=True, choices=[
        'research', 'script', 'audio', 'timestamps', 'visual', 'video', 'publish'
    ])
    p_stage.add_argument('--topic-slug', required=True, help='Topic slug')
    p_stage.add_argument('--topic', help='Topic override (for research stage)')
    p_stage.add_argument('--force', action='store_true', help='Force re-run even if completed')
    p_stage.set_defaults(func=cmd_stage)
    
    # Status
    p_status = subparsers.add_parser('status', help='Show pipeline status')
    p_status.add_argument('--topic-slug', required=True, help='Topic slug')
    p_status.set_defaults(func=cmd_status)
    
    # Topic management
    p_topic = subparsers.add_parser('topic', help='Topic management')
    p_topic.add_argument('--list', action='store_true', help='List all topics')
    p_topic.add_argument('--next', action='store_true', help='Show next topic')
    p_topic.set_defaults(func=cmd_topic)
    
    # Clean
    p_clean = subparsers.add_parser('clean', help='Clean output directory')
    p_clean.add_argument('--topic-slug', help='Specific topic to clean (default: all)')
    p_clean.set_defaults(func=cmd_clean)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_cli_logging(args.verbose)
    
    # Load config
    if args.config:
        reload_config(args.config)
    
    # Run command
    try:
        return args.func(args)
    except SleepHistoryError as e:
        logging.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
