#!/usr/bin/env python3
"""Sleep History Bot - Main CLI Entry Point.

Zero-cost automated History for Sleep YouTube channel production.
"""
import sys
import argparse
import logging
import json
import shutil
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
            shutil.rmtree(topic_dir)
            print(f"Cleaned: {topic_dir}")
        else:
            print(f"Not found: {topic_dir}")
    else:
        # Clean all
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Cleaned all: {output_dir}")
        else:
            print(f"Not found: {output_dir}")
    
    return 0


def cmd_plan(args):
    """Generate plan (prompts list) for manual image creation - like old 'plan' mode."""
    config_path = args.config
    if config_path:
        reload_config(config_path)
    
    from src.stages.research import run_research
    from src.stages.script import run_script
    from src.stages.audio import run_audio
    from src.stages.timestamps import run_timestamps
    from src.stages.visual import run_visual
    
    orchestrator = create_orchestrator(Path(args.output), config_path)
    
    # Run up to visual stage
    result = orchestrator.run_full_pipeline(
        topic_override=args.topic,
        stages=['research', 'script', 'audio', 'timestamps', 'visual']
    )
    
    if not result.success:
        print(f"❌ Plan generation failed at: {result.failed_stage}")
        return 1
    
    # Load visual checkpoint to get scenes
    topic_slug = result.topic_slug
    visual_checkpoint = None
    cp_mgr = None
    from src.core.checkpoint import create_checkpoint_manager
    cp_mgr = create_checkpoint_manager(Path(args.output), topic_slug)
    visual_checkpoint = cp_mgr.load('visual')
    
    if not visual_checkpoint:
        print("❌ No visual checkpoint found")
        return 1
    
    # Generate plan output
    from src.stages.visual import VisualStage
    from src.services.gemini import get_gemini_client
    
    gemini = get_gemini_client()
    script_cp = cp_mgr.load('script')
    timestamps_cp = cp_mgr.load('timestamps')
    audio_cp = cp_mgr.load('audio')
    
    topic = result.topic
    duration_min = int(audio_cp.get('duration_seconds', 0) / 60)
    scenes = visual_checkpoint.get('scenes', [])
    
    # Generate plan text
    plan_lines = [
        f"# {topic}",
        f"{duration_min} min video, {len(scenes)} scenes.\n",
        "HOW TO USE",
        "1. For each scene below, paste the IMAGE PROMPT into an AI image generator and save the picture.",
        "2. Name the file with the scene number: 001.png, 002.png, 003.png ...",
        "3. For scenes marked VIDEO, upload your picture to an image-to-video tool, paste the VIDEO PROMPT, save the clip as 001.mp4 (same number).",
        "4. Place everything in the 'media' folder of your GitHub repo.",
        f"5. Thumbnail: make it yourself and name it thumbnail.png\n",
        f"MINIMUM: the VIDEO scenes ({len([s for s in scenes if s['index'] % 3 == 0])} of them):",
    ]
    
    # Add video scene numbers
    video_scenes = [f"{s['index']+1:03d}" for s in scenes if s['index'] % 3 == 0]
    plan_lines.append(", ".join(video_scenes[:40]) + (" ..." if len(video_scenes) > 40 else ""))
    plan_lines.append("")
    
    for scene in scenes:
        n = scene['index'] + 1
        plan_lines.append(f"## {n:03d}  {scene['start']:.0f}-{scene['end']:.0f}s  [{scene['style']}]" + ("  VIDEO" if scene['index'] % 3 == 0 else ""))
        plan_lines.append(f"Narration: {scene['text'][:220]}{'...' if len(scene['text']) > 220 else ''}")
        plan_lines.append(f"IMAGE PROMPT: {scene['prompt']}")
        if scene['index'] % 3 == 0:
            # Get motion prompt
            from src.stages.visual import VisualStage
            stage = VisualStage(Path(args.output), topic_slug)
            plan_lines.append(f"VIDEO PROMPT: {stage.get_motion_prompt(VisualStyle(scene['style']))} (5-8 seconds, 16:9)")
        plan_lines.append("")
    
    plan_text = "\n".join(plan_lines)
    plan_path = Path(args.output) / topic_slug / "plan" / "PLAN.txt"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan_text)
    
    print(f"\n✅ Plan generated: {plan_path}")
    print(f"   Topic: {topic}")
    print(f"   Scenes: {len(scenes)}")
    print(f"   Duration: ~{duration_min} min")
    return 0


def cmd_assemble(args):
    """Assemble video from user-uploaded media - like old 'assemble' mode."""
    config_path = args.config
    if config_path:
        reload_config(config_path)
    
    from src.core.checkpoint import create_checkpoint_manager
    from src.stages.video import run_video
    from src.stages.publish import run_publish
    
    # Need to find the plan directory
    plan_dir = Path(args.plan_dir) if args.plan_dir else Path("plan_in")
    
    if not plan_dir.exists():
        print(f"❌ Plan directory not found: {plan_dir}")
        print("   Run with --plan-dir pointing to the plan output from a previous run")
        return 1
    
    # Load plan metadata
    plan_json = plan_dir / "plan.json"
    if not plan_json.exists():
        print(f"❌ plan.json not found in {plan_dir}")
        return 1
    
    plan_data = json.loads(plan_json.read_text())
    topic = plan_data['topic']
    scenes = plan_data['scenes']
    duration_min = plan_data.get('minutes', 22)
    topic_slug = slugify(topic)
    
    output_dir = Path(args.output)
    topic_dir = output_dir / topic_slug
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy subtitles
    subs_src = plan_dir / "subs.srt"
    subs_dst = topic_dir / "subs.srt"
    if subs_src.exists():
        shutil.copy2(subs_src, subs_dst)
    
    # Collect user media
    media_dir = Path("media")
    imgs, vids, thumbs = {}, {}, []
    if media_dir.exists():
        for p in sorted(media_dir.glob("*")):
            if p.stem.lower().startswith("thumb"):
                thumbs.append(p)
                continue
            # Extract number from filename
            import re
            m = re.search(r"(\d+)", p.stem)
            if m:
                n = int(m.group(1))
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    imgs[n] = p
                elif p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"):
                    vids[n] = p
    
    print(f"TOPIC: {topic} | {len(scenes)} scenes | Your media: {len(imgs)} images, {len(vids)} clips, {len(thumbs)} thumbnails")
    
    # Prepare scenes with user media
    for scene in scenes:
        n = scene['index'] + 1
        img_path = topic_dir / f"img_{scene['index']:03d}.png"
        
        if n in imgs:
            # Use user image
            shutil.copy2(imgs[n], img_path)
            scene['image_path'] = str(img_path)
        
        if n in vids:
            # Use user video clip
            cap = cv2.VideoCapture(str(vids[n]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not img_path.exists():
                ok, f = cap.read()
                if ok:
                    cv2.imwrite(str(img_path), f)
            cap.release()
            if frames and frames / fps > 1.0:
                scene['clip'] = {"path": str(vids[n]), "dur": frames / fps}
                scene['image_path'] = str(img_path)
    
    # Need audio from plan
    narration_mp3 = plan_dir / "narration.mp3"
    if not narration_mp3.exists():
        print("❌ narration.mp3 not found in plan directory")
        return 1
    
    # Convert to wav if needed
    narration_wav = topic_dir / "narration.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", str(narration_mp3), "-ac", "1", "-ar", "24000", str(narration_wav)], check=True)
    
    # Create minimal checkpoints for video stage
    visual_cp = {"scenes": scenes}
    audio_cp = {"narration_path": str(narration_wav), "duration_seconds": duration_min * 60}
    timestamps_cp = {"srt_path": str(subs_dst) if subs_dst.exists() else ""}
    
    # Run video rendering
    try:
        video_cp = run_video(output_dir, topic_slug, visual_cp, audio_cp, timestamps_cp)
        print(f"✅ Video rendered: {video_cp.video_path}")
    except Exception as e:
        print(f"❌ Video rendering failed: {e}")
        return 1
    
    # Run publish stage
    try:
        # Need script and research checkpoints - create minimal ones
        script_cp = {"topic": topic, "outline": [], "sections": [], "full_script": ""}
        research_cp = {"topic": topic}
        
        publish_cp = run_publish(
            output_dir, topic_slug, video_cp.__dict__ if hasattr(video_cp, '__dict__') else video_cp,
            audio_cp, timestamps_cp, visual_cp, script_cp, research_cp
        )
        print(f"✅ Publish package ready: {publish_cp.video_path}")
        print(f"   Approval ready: {publish_cp.approval_ready}")
    except Exception as e:
        print(f"⚠️  Publish stage failed (video still rendered): {e}")
    
    return 0


def cmd_thumbnail(args):
    """Generate thumbnails using Gemini (optional)."""
    config_path = args.config
    if config_path:
        reload_config(config_path)
    
    from src.services.gemini import get_gemini_client
    from PIL import Image
    
    gemini = get_gemini_client()
    topic = args.topic
    minutes = args.minutes
    variant = args.variant
    output_path = Path(args.output)
    
    looks = [
        "a wide, dramatic establishing view of the place with its most famous landmark",
        "a warm street-level view with glowing lanterns and a few people walking",
        "a cozy peaceful view through a window with a candle and the city beyond"
    ]
    
    prompt = (f'Create a premium 16:9 YouTube thumbnail for a calm history sleep story. '
              f'Scene: {looks[variant]} for "{topic}", '
              f'at night with a glowing full moon, deep navy sky, warm golden lantern light, '
              f'rich cinematic detail. '
              f'Add elegant large golden serif title text "{topic}" (the last words bigger), '
              f'and a small top badge with a crescent moon icon and the words "{minutes} MIN SLEEP STORY". '
              f'Text must be spelled exactly, sharp and readable. '
              f'High contrast, polished, no watermark.')
    
    try:
        image_data = gemini.generate_image(prompt)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"✅ Thumbnail generated: {output_path}")
        return 0
    except Exception as e:
        print(f"❌ Thumbnail generation failed: {e}")
        return 1


# Need slugify for assemble
from src.utils.text import slugify
import cv2


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
  python main.py stage --stage script --topic-slug my-topic
  
  # Check status
  python main.py status --topic-slug my-topic
  
  # List topics
  python main.py topic --list
  
  # Get next topic
  python main.py topic --next
  
  # Generate plan (prompts for manual creation)
  python main.py plan --topic "Ancient Rome Jobs"
  
  # Assemble from user media
  python main.py assemble --plan-dir plan_in
  
  # Generate thumbnail with Gemini
  python main.py thumbnail --topic "Ancient Rome" --minutes 22 --variant 0 --output thumb.jpg
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
    
    # Plan mode
    p_plan = subparsers.add_parser('plan', help='Generate plan (prompts list) for manual creation')
    p_plan.add_argument('--topic', help='Override topic (default: next from topics.txt)')
    p_plan.set_defaults(func=cmd_plan)
    
    # Assemble mode
    p_assemble = subparsers.add_parser('assemble', help='Assemble video from user media')
    p_assemble.add_argument('--plan-dir', help='Plan directory from previous run')
    p_assemble.set_defaults(func=cmd_assemble)
    
    # Thumbnail generation
    p_thumb = subparsers.add_parser('thumbnail', help='Generate thumbnail with Gemini')
    p_thumb.add_argument('--topic', required=True, help='Video topic')
    p_thumb.add_argument('--minutes', type=int, required=True, help='Video duration in minutes')
    p_thumb.add_argument('--variant', type=int, default=0, choices=[0,1,2], help='Thumbnail variant')
    p_thumb.add_argument('--output', required=True, help='Output path')
    p_thumb.set_defaults(func=cmd_thumbnail)
    
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
