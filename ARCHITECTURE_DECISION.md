# Architecture Decision: Old vs New Implementation

## Old Monolithic (make_video.py) - Strengths
- ✅ **Advanced video rendering**: 3D parallax depth (MiDaS), Ken Burns with zoom+pan, ember particles, vignette, film grain
- ✅ **B-roll video clip support**: Can incorporate user-uploaded video clips
- ✅ **Assemble mode**: Build video from user media
- ✅ **Plan mode**: Generate prompt lists for manual creation
- ✅ **Subtitle burn-in**: Can render subtitles into video
- ✅ **Logo overlay**: Channel watermark
- ✅ **Multiple Gemini models**: Tries multiple models with fallback
- ✅ **Sophisticated prompts**: Detailed image prompts per style
- ✅ **Production-proven**: Has actually generated videos

## New Modular (src/) - Strengths
- ✅ **Modularity**: Run individual stages independently
- ✅ **Checkpoint/resume**: Persist progress at each stage
- ✅ **Configuration**: YAML + env overrides, single source of truth
- ✅ **Quality gates**: Automated validation (duration, audio, video, content)
- ✅ **Structured logging**: Rotating files, multiple loggers
- ✅ **Type safety**: Pydantic models for all data
- ✅ **Error handling**: Custom exception hierarchy with context
- ✅ **CLI**: Full command set (full, stage, status, topic, clean)
- ✅ **Research stage**: Source gathering, fact-checking, deduplication
- ✅ **Cache system**: Generic + image-specific caching
- ✅ **Tests**: Unit test structure
- ✅ **GitHub Actions**: Modular workflow with stage-level control

## Decision: Hybrid Architecture
**Keep modular architecture as foundation, port advanced video features from old implementation.**

### Components to PORT from old → new:
1. **VideoStage** - Add 3D depth, embers, vignette, film grain, b-roll, logo, subtitle burn-in
2. **VisualStage** - Add sophisticated prompt engineering, style-specific parameters
3. **Gemini service** - Add multi-model fallback logic
4. **Thumbnail** - Add Gemini thumbnail generation option
5. **Assemble/Plan modes** - Add as CLI options

### Components to REMOVE:
- `make_video.py` (legacy monolithic - replaced by modular system)

### Files to DELETE after porting:
- `make_video.py` - 1211 lines of legacy code

### Implementation Plan:
1. Enhance `src/stages/video.py` with all old rendering features
2. Enhance `src/stages/visual.py` with advanced prompting
3. Enhance `src/services/gemini.py` with model fallback
4. Add assemble/plan CLI commands
5. Add quality gates for new video features
6. Remove `make_video.py`
7. Test end-to-end
