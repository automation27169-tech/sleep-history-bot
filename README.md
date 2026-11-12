# Sleep History Bot

A **zero-cost, fully automated** YouTube channel production system for "History for Sleep" videos. Generates professional 15-30 minute historical storytelling videos with calm narration, historical visuals, and professional editing — all running on free infrastructure.

## Features

- 🎬 **Complete Pipeline**: Topic → Research → Script → Audio → Visuals → Video → Publish
- 💰 **Zero Cost**: Uses only free tiers (Gemini API, Pollinations.ai, Edge-TTS, GitHub Actions)
- 🔄 **Resumable**: Checkpoint system saves progress at each stage
- 🛡️ **Quality Gates**: Automated validation of duration, audio, video, and content
- 📦 **Approval Package**: Generates thumbnails, metadata, and quality report for manual review
- ☁️ **GitHub Actions**: Run on free GitHub runners (2000 min/month)

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  RESEARCH   │──▶│   SCRIPT    │──▶│   AUDIO     │──▶│ TIMESTAMPS  │
│  (topics,   │   │  (outline,  │   │  (Gemini    │   │  (Whisper   │
│   sources)  │   │   sections) │   │   TTS/Edge) │   │   align)    │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                                                              │
┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  PUBLISH    │◀──│   VIDEO     │◀──│   VISUAL    │──────────┘
│  (meta,     │   │  (FFmpeg,   │   │  (Pollinat- │
│   quality)  │   │   Ken Burns)│   │   ions)     │
└─────────────┘   └─────────────┘   └─────────────┘
```

## Quick Start

### 1. Prerequisites

- GitHub repository (public for free Actions)
- **GEMINI_API_KEY** - Get free at [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Setup

```bash
# Clone repository
git clone https://github.com/yourusername/sleep-history-bot
cd sleep-history-bot

# Add Gemini API key as GitHub Secret
# Settings → Secrets → Actions → New secret: GEMINI_API_KEY
```

### 3. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt
sudo apt-get install ffmpeg fonts-dejavu-core

# Run full pipeline (auto-selects next topic)
python main.py full

# Or specify a topic
python main.py full --topic "The Strangest Jobs in Ancient Rome"

# Resume from a specific stage
python main.py full --resume-from video
```

### 4. Run on GitHub Actions

Go to **Actions → Make Video → Run workflow** and select:
- **Mode**: `full` (complete pipeline) or individual stage
- **Topic**: Optional override
- **Resume from**: Stage to resume from (if previous run failed)

## Configuration

All settings in `config/settings.yaml`:

```yaml
channel:
  target_duration_min: 22  # Target video length
  
script:
  sections: 8              # Number of script sections
  words_per_section: 700   # Words per section
  
audio:
  tts_primary: "gemini"    # Primary TTS
  gemini_voice: "Algieba"  # Voice name
  
visual:
  image_seconds: 25        # Seconds per image
  fps: 24                  # Video FPS
  
quality:
  min_duration_seconds: 900   # 15 min minimum
  max_duration_seconds: 1800  # 30 max maximum
```

## CLI Commands

```bash
# Full pipeline
python main.py full [--topic TOPIC] [--resume-from STAGE] [--stages STAGE1,STAGE2]

# Single stage
python main.py stage --stage SCRIPT --topic-slug my-topic

# Status
python main.py status --topic-slug my-topic

# Topic management
python main.py topic --list      # List all topics
python main.py topic --next      # Show next topic

# Cleanup
python main.py clean [--topic-slug SLUG]
```

## Output Structure

```
output/topic-slug/
├── checkpoints/           # JSON checkpoints per stage
│   ├── research.json
│   ├── script.json
│   ├── audio.json
│   ├── timestamps.json
│   ├── visual.json
│   ├── video.json
│   └── publish.json
├── narration.wav          # Final audio
├── subs.srt              # Subtitles
├── img_000.png ...       # Scene images
├── thumbnail_a.jpg       # Thumbnail variants
├── thumbnail_b.jpg
├── thumbnail_c.jpg
├── video.mp4             # Final video
└── metadata.json         # Publishing package
```

## Publishing Workflow

1. **Run pipeline** (GitHub Actions or local)
2. **Download `output` artifact** from Actions
3. **Review `metadata.json`** - contains:
   - `title`, `description`, `tags`, `chapters`
   - `quality_report` with pass/fail status
   - `thumbnails` paths
   - `video_path`
4. **Upload manually** to YouTube Studio
5. **Mark topic done** - automatically committed to `done.txt`

## Zero-Cost Services Used

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Google Gemini API | Script, TTS, Images | 1500 req/day |
| Pollinations.ai | Image generation | Unlimited* |
| Edge-TTS | TTS fallback | Unlimited |
| Faster-Whisper | Word alignment | Local CPU |
| FFmpeg | Video rendering | Local |
| GitHub Actions | Automation | 2000 min/mo |

*Soft limits apply, no hard quota published

## Project Structure

```
sleep-history-bot/
├── config/
│   ├── settings.yaml      # All configuration
│   └── logging.yaml       # Logging config
├── src/
│   ├── core/              # Config, checkpoints, logging, models
│   ├── pipeline/          # Orchestrator, stage runner, quality gates
│   ├── stages/            # 7 pipeline stages
│   ├── services/          # External API wrappers
│   └── utils/             # Text, video, audio utilities
├── tests/                 # Unit tests
├── data/
│   ├── topics.txt         # Topic queue
│   └── done.txt           # Completed topics
├── .github/workflows/     # GitHub Actions
├── main.py                # CLI entry point
└── requirements.txt
```

## Requirements

- Python 3.11+
- FFmpeg
- GitHub account (for Actions)

See `requirements.txt` for Python packages.

## License

MIT License - Feel free to use and modify.

## Disclaimer

This tool generates AI-created content. Always review for historical accuracy before publishing. The system includes fact-checking but cannot guarantee 100% accuracy.
