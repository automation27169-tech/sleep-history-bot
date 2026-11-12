# IMPLEMENTATION PLAN
## Sleep History Bot - Modular Zero-Cost Architecture

---

## PROJECT STRUCTURE

```
sleep-history-bot/
├── config/
│   ├── settings.yaml          # All tunable parameters
│   └── logging.yaml           # Logging configuration
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration loader
│   │   ├── checkpoint.py      # Checkpoint/resume system
│   │   ├── logging.py         # Structured logging
│   │   ├── models.py          # Data classes (Pydantic)
│   │   └── exceptions.py      # Custom exceptions
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # Main pipeline runner
│   │   ├── stage_runner.py    # Stage execution with retry
│   │   └── quality_gates.py   # Validation between stages
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── research.py        # Topic selection, fact-checking
│   │   ├── script.py          # Outline, sections, review
│   │   ├── audio.py           # TTS, loudnorm, duration
│   │   ├── timestamps.py      # Whisper alignment, SRT
│   │   ├── visual.py          # Scenes, prompts, images
│   │   ├── video.py           # Rendering, effects, encoding
│   │   └── publish.py         # Thumbnails, metadata, package
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gemini.py          # Gemini API wrapper
│   │   ├── pollinations.py    # Pollinations image generation
│   │   ├── edge_tts.py        # Edge-TTS wrapper
│   │   ├── whisper.py         # Faster-Whisper alignment
│   │   ├── ffmpeg.py          # FFmpeg wrapper
│   │   └── cache.py           # File-based caching
│   └── utils/
│       ├── __init__.py
│       ├── text.py            # Text processing utilities
│       ├── video.py           # Video utilities
│       └── audio.py           # Audio utilities
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_checkpoint.py
│   ├── test_research.py
│   ├── test_script.py
│   └── test_integration.py
├── data/
│   ├── topics.txt             # Topic queue
│   ├── done.txt               # Completed topics
│   └── cache/                 # Persistent cache (gitignored)
├── output/                    # Generated videos (gitignored)
├── media/                     # User uploads for assemble mode (gitignored)
├── plans/                     # Plan artifacts (gitignored)
├── logs/                      # Runtime logs (gitignored)
├── .github/
│   └── workflows/
│       └── make_video.yml     # Modular workflow
├── main.py                    # CLI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## STAGE SPECIFICATIONS

### Stage 1: Research (`research.py`)
**Input**: None (reads topics.txt, done.txt)
**Output**: `checkpoints/research.json`
```json
{
  "topic": "The Strangest Jobs in Ancient Rome",
  "topic_slug": "the-strangest-jobs-in-ancient-rome",
  "sources": [
    {"title": "Roman Occupations", "url": "...", "credibility": 0.9},
    ...
  ],
  "fact_checks": [
    {"claim": "Urine was used for laundry", "verified": true, "source": 0},
    ...
  ],
  "duplicate_check": false
}
```

**Functions**:
- `select_next_topic()` - picks from topics.txt, skips done.txt
- `verify_topic_unique()` - fuzzy match against done.txt
- `fetch_sources(topic)` - web search via Gemini (or local knowledge)
- `fact_check_claims(script)` - verify dates, names, events

---

### Stage 2: Script (`script.py`)
**Input**: `research.json`
**Output**: `checkpoints/script.json`
```json
{
  "outline": [
    {"title": "Introduction", "summary": "...", "target_words": 800},
    ...
  ],
  "sections": [
    {"index": 1, "title": "...", "text": "...", "word_count": 812},
    ...
  ],
  "full_script": "Complete narration text...",
  "total_words": 6400,
  "estimated_duration_min": 22.5,
  "review": {
    "accuracy_score": 0.95,
    "repetition_score": 0.02,
    "pacing_score": 0.9,
    "issues": []
  }
}
```

**Functions**:
- `generate_outline(topic, sources, sections=8)` - structured outline
- `write_section(outline, index, previous_tail)` - individual section
- `review_script(full_script)` - accuracy, repetition, pacing, grammar
- `estimate_duration(words, wpm=140)` - target 140 wpm for sleep narration

---

### Stage 3: Audio (`audio.py`)
**Input**: `script.json`
**Output**: `checkpoints/audio.json`
```json
{
  "narration_path": "output/topic/narration.wav",
  "duration_seconds": 1350.5,
  "chunks": [
    {"index": 0, "path": "voice_000.wav", "text": "...", "duration": 45.2},
    ...
  ],
  "loudnorm_applied": true,
  "sample_rate": 24000,
  "channels": 1
}
```

**Functions**:
- `chunk_script(script, max_chars=2500)` - split for TTS limits
- `tts_gemini(chunks, voice="Algieba")` - primary TTS
- `tts_edge(chunks, voice="en-US-GuyNeural")` - fallback
- `concat_and_loudnorm(chunk_paths, output_path)` - FFmpeg
- `verify_audio(path)` - duration, silence detection, clipping check

---

### Stage 4: Timestamps (`timestamps.py`)
**Input**: `script.json`, `audio.json`
**Output**: `checkpoints/timestamps.json`
```json
{
  "word_timestamps": [
    {"word": "Welcome", "start": 0.0, "end": 0.45},
    ...
  ],
  "sentences": [
    {"text": "Welcome to tonight's story.", "start": 0.0, "end": 2.1},
    ...
  ],
  "srt_path": "output/topic/subs.srt",
  "alignment_method": "whisper",  // or "estimated"
  "alignment_quality": 0.87
}
```

**Functions**:
- `split_sentences(script)` - sentence boundaries with word indices
- `whisper_align(audio_path)` - word-level timestamps
- `align_script_to_whisper(script_words, whisper_words)` - difflib matching
- `generate_srt(sentences, timestamps)` - SRT file

---

### Stage 5: Visual (`visual.py`)
**Input**: `timestamps.json`, `script.json`
**Output**: `checkpoints/visual.json`
```json
{
  "scenes": [
    {
      "index": 0,
      "start": 0.0,
      "end": 25.0,
      "text": "Welcome to tonight's story...",
      "prompt": "Ancient Roman street at night, lanterns glowing...",
      "style": "cinematic",
      "image_path": "output/topic/img_000.png"
    },
    ...
  ],
  "thumbnails": [
    {"variant": 0, "path": "output/topic/thumbnail_a.jpg"},
    {"variant": 1, "path": "output/topic/thumbnail_b.jpg"},
    {"variant": 2, "path": "output/topic/thumbnail_c.jpg"}
  ]
}
```

**Functions**:
- `build_scenes(sentences, image_seconds=25)` - group by time/topic
- `generate_prompts(topic, scenes)` - LLM prompts for each scene
- `generate_images(prompts, cache_dir)` - Pollinations with caching
- `generate_thumbnails(topic, images, duration_min)` - PIL variants

---

### Stage 6: Video (`video.py`)
**Input**: `visual.json`, `audio.json`
**Output**: `checkpoints/video.json`
```json
{
  "video_path": "output/topic/video.mp4",
  "duration_seconds": 1350.5,
  "resolution": "1920x1080",
  "fps": 24,
  "codec": "libx264",
  "crf": 23,
  "bitrate_kbps": 3500,
  "audio_codec": "aac",
  "audio_bitrate": 160,
  "has_subtitles": true,
  "has_ambience": true,
  "render_time_seconds": 420
}
```

**Functions**:
- `build_shots(scenes, duration, broll_pool)` - shot list with transitions
- `render_frame(shot, progress)` - Ken Burns, 3D parallax, embers
- `encode_video(frames, audio_path, output_path)` - FFmpeg pipeline
- `verify_video(path)` - duration, resolution, audio sync, black frames

---

### Stage 7: Publish (`publish.py`)
**Input**: `video.json`, `script.json`, `visual.json`
**Output**: `checkpoints/publish.json` (approval package)
```json
{
  "video_path": "output/topic/video.mp4",
  "thumbnails": ["thumbnail_a.jpg", "thumbnail_b.jpg", "thumbnail_c.jpg"],
  "metadata": {
    "title": "The Strangest Jobs in Ancient Rome | Relaxing History for Sleep",
    "description": "Tonight we explore... [3-4 sentences]\n\nChapters:\n0:00 Introduction\n...",
    "tags": ["history", "sleep", "ancient rome", "relaxing", "documentary", ...],
    "chapters": ["0:00 Introduction", "3:22 Urine Collectors", ...]
  },
  "quality_report": {
    "duration_check": {"target": "15-30min", "actual": "22:30", "pass": true},
    "audio_check": {"loudness": -16, "clipping": false, "silence_gaps": 0, "pass": true},
    "video_check": {"resolution": "1920x1080", "black_frames": 0, "sync_ok": true, "pass": true},
    "content_check": {"accuracy": 0.95, "originality": 0.98, "pass": true},
    "overall_pass": true
  },
  "approval_ready": true
}
```

**Functions**:
- `generate_metadata(topic, outline, chapters)` - AI title/description/tags
- `create_approval_package(video_path, thumbnails, metadata, quality_report)`
- `verify_quality(video_path, audio_path, script)` - all quality gates

---

## CONFIGURATION (settings.yaml)

```yaml
# Channel settings
channel:
  name: "Sleep History"
  language: "en"
  target_duration_min: 22
  min_duration_min: 15
  max_duration_min: 30

# Topic settings
topics:
  file: "data/topics.txt"
  done_file: "data/done.txt"
  similarity_threshold: 0.85  # for duplicate detection

# Script settings
script:
  sections: 8
  words_per_section: 700
  story_style: "documentary"  # or "story" (second-person)
  welcome_section: true
  fade_out_section: true

# Audio settings
audio:
  tts_primary: "gemini"
  tts_fallback: "edge"
  gemini_voice: "Algieba"
  edge_voice: "en-US-GuyNeural"
  narration_style: "very slow, soft, warm, calm and soothing, with long gentle pauses"
  target_loudness: -16  # LUFS
  target_lra: 7
  target_tp: -1.5
  padding_end_seconds: 6
  chunk_size_chars: 2500

# Timestamp settings
timestamps:
  whisper_model: "base.en"
  whisper_device: "cpu"
  whisper_compute_type: "int8"
  min_alignment_ratio: 0.6
  fallback_to_estimated: true

# Visual settings
visual:
  image_seconds: 25
  shot_seconds: 9
  fps: 24
  resolution: [1920, 1080]
  crop_right: 0.15
  crop_bottom: 0.06
  crossfade_seconds: 1.2
  cut_fade_seconds: 0.35
  styles:
    cinematic: "dark moody cinematic photoreal film still..."
    manuscript: "medieval illuminated manuscript style illustration..."
    engraving: "black and white 19th century etching engraving..."
  broll_every: 3
  embers_enabled: true
  three_d_enabled: true

# Image generation
images:
  primary: "pollinations"
  fallback: "gemini"
  pollinations_base_url: "https://image.pollinations.ai/prompt/"
  cache_enabled: true
  cache_dir: "data/cache/images"

# Video encoding
video:
  codec: "libx264"
  preset: "veryfast"
  crf: 23
  maxrate: "3500k"
  bufsize: "7M"
  colorspace: "bt709"
  audio_codec: "aac"
  audio_bitrate: "160k"

# Thumbnails
thumbnails:
  resolution: [1280, 720]
  variants: 3
  font: "DejaVuSans-Bold.ttf"

# Quality gates
quality:
  min_duration_seconds: 900
  max_duration_seconds: 1800
  max_black_frames: 5
  max_silence_gaps: 3
  min_loudness_lufs: -18
  max_loudness_lufs: -14
  min_accuracy_score: 0.9
  min_originality_score: 0.95

# Pipeline
pipeline:
  checkpoint_dir: "output/{topic_slug}/checkpoints"
  cleanup_temp: true
  retry_attempts: 3
  retry_base_delay: 15
  max_retry_delay: 180
```

---

## GITHUB ACTIONS WORKFLOW (Modular)

```yaml
name: Make Video

on:
  workflow_dispatch:
    inputs:
      mode:
        type: choice
        options: [full, research, script, audio, timestamps, visual, video, publish]
        default: full
      topic:
        type: string
        required: false
      resume_from:
        type: string
        required: false
      dry_run:
        type: boolean
        default: false

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 300
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install system deps
        run: |
          sudo apt-get update
          sudo apt-get install -y ffmpeg fonts-dejavu-core
      - name: Install Python deps
        run: pip install -r requirements.txt
      - name: Install ML deps (for video stage)
        if: inputs.mode != 'research' && inputs.mode != 'script'
        continue-on-error: true
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install transformers
      - name: Restore checkpoint
        if: inputs.resume_from
        uses: actions/download-artifact@v4
        with:
          name: checkpoint-${{ inputs.resume_from }}
          path: output/
      - name: Run Pipeline
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          TOPIC: ${{ inputs.topic }}
          MODE: ${{ inputs.mode }}
          DRY_RUN: ${{ inputs.dry_run }}
        run: |
          python main.py --mode ${{ inputs.mode }} --topic "${{ inputs.topic }}" \
            ${DRY_RUN && echo '--dry-run'} ${RESUME_FROM && echo "--resume-from ${{ inputs.resume_from }}"}
      - name: Upload checkpoint
        uses: actions/upload-artifact@v4
        with:
          name: checkpoint-${{ github.run_id }}
          path: output/**/checkpoints/
          retention-days: 30
      - name: Upload output
        if: inputs.mode == 'full' || inputs.mode == 'video' || inputs.mode == 'publish'
        uses: actions/upload-artifact@v4
        with:
          name: output
          path: output/
          retention-days: 30
      - name: Save progress
        if: inputs.mode == 'full' || inputs.mode == 'publish'
        run: |
          git config user.name "bot"
          git config user.email "bot@users.noreply.github.com"
          git add data/done.txt
          git diff --cached --quiet || (git commit -m "progress: ${{ inputs.topic }}" && git push)
```

---

## IMPLEMENTATION ORDER (Dependencies)

```
1. config/ (settings.yaml, logging.yaml)
2. src/core/ (config, checkpoint, logging, models, exceptions)
3. src/services/ (gemini, pollinations, edge_tts, whisper, ffmpeg, cache)
4. src/utils/ (text, video, audio)
5. src/stages/research.py
6. src/stages/script.py
7. src/stages/audio.py
8. src/stages/timestamps.py
9. src/stages/visual.py
10. src/stages/video.py
11. src/stages/publish.py
12. src/pipeline/ (orchestrator, stage_runner, quality_gates)
13. main.py (CLI)
14. tests/
15. .github/workflows/make_video.yml
16. README.md, .gitignore
```

---

## TESTING STRATEGY

### Unit Tests (pytest)
- Config loading/validation
- Checkpoint save/load/resume
- Topic selection/deduplication
- Text chunking/sentence splitting
- Word alignment algorithm
- Scene building logic
- SRT generation

### Integration Tests (require API keys)
- Full pipeline dry-run (no API calls)
- Single stage execution with mocks
- GitHub Actions workflow syntax

### Quality Tests (post-render)
- Video duration verification
- Audio loudness measurement
- Subtitle sync check
- Black frame detection

---

## ROLLBACK/RECOVERY

Each stage produces a checkpoint. On failure:
1. Fix the issue
2. Re-run with `--resume-from <run_id>`
3. Pipeline loads last successful checkpoint
4. Continues from failed stage

Checkpoint structure:
```
output/topic-slug/
├── checkpoints/
│   ├── research.json
│   ├── script.json
│   ├── audio.json
│   ├── timestamps.json
│   ├── visual.json
│   ├── video.json
│   └── publish.json
├── narration.wav
├── subs.srt
├── img_000.png ... img_N.png
├── thumbnail_a.jpg, thumbnail_b.jpg, thumbnail_c.jpg
├── video.mp4
└── metadata.json
```

---

## ESTIMATED EFFORT

| Phase | Tasks | Est. Days |
|-------|-------|-----------|
| Foundation | Structure, config, checkpoint, logging | 2 |
| Core Services | Gemini, Pollinations, Edge-TTS, Whisper, FFmpeg wrappers | 3 |
| Pipeline Stages | 7 stages × 1 day each | 7 |
| Orchestration | Runner, quality gates, CLI | 2 |
| Testing | Unit + integration + quality | 3 |
| Workflow | GitHub Actions, documentation | 2 |
| **Total** | | **~19 days** |

---

*Plan created: 2026-11-12*
*Status: Ready for implementation*
