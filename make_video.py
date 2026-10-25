#!/usr/bin/env python3
"""Topic -> script -> voice -> images -> video. Free stack: Gemini + Pollinations + FFmpeg."""
import os, sys, re, json, textwrap, time, wave, math, random, shutil, asyncio, subprocess, difflib, urllib.parse
from pathlib import Path
import numpy as np
import cv2
import requests
from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-flash-latest")
TTS_MODEL = os.getenv("TTS_MODEL", "")   # optional override; otherwise TTS_MODELS below are tried in order
VOICE = os.getenv("VOICE", "Algieba")              # one fixed voice for the whole channel
EDGE_VOICE = os.getenv("EDGE_VOICE", "en-US-GuyNeural")
STORY_STYLE = os.getenv("STORY_STYLE", "documentary").lower()   # documentary | story (second-person "You walk...")
SECTIONS = int(os.getenv("SECTIONS", "8"))          # 8 sections ~ 45-50 min of audio
WORDS_PER_SECTION = int(os.getenv("WORDS_PER_SECTION", "800"))
IMAGE_SECONDS = int(os.getenv("IMAGE_SECONDS", "40" if os.getenv("MODE", "auto").lower() == "plan" else "25"))
FPS = 24
OUT_W, OUT_H = 1920, 1080                           # Full HD (real 4K is not possible with free generators)
CROP_RIGHT = float(os.getenv("CROP_RIGHT", "0.15"))   # cut this share off the right/bottom edge: removes the free-service watermark
CROP_BOTTOM = float(os.getenv("CROP_BOTTOM", "0.06"))
FADE = 1.2                                          # crossfade seconds between different images
CUT_FADE = 0.35                                     # quick dissolve between shots of the same image
SHOT_SECONDS = float(os.getenv("SHOT_SECONDS", "9"))   # every picture is shown in several short shots (reference channel ~8 s)
EMBERS = os.getenv("EMBERS", "on").lower() != "off"
THREE_D = os.getenv("THREE_D", "on").lower() != "off"

BROLL_EVERY = int(os.getenv("BROLL_EVERY", "3"))          # put one of your short video clips before every Nth picture
MODE = os.getenv("MODE", "auto").lower()                 # plan | assemble | auto
BURN_SUBS = os.getenv("SUBTITLES", "off").lower() == "on"   # subs.srt is always written for YouTube captions
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
STYLE = os.getenv("STYLE", "cinematic, ultra detailed, dramatic moonlit night, warm golden lantern and torch light "
                           "against a deep navy blue sky, glowing full moon, volumetric light, rich textures, "
                           "photorealistic digital painting, wide composition, no text, no watermark")
NARRATION_STYLE = ("Read this as a sleep story narrator: very slow, soft, warm, calm and soothing, "
                   "with long gentle pauses between sentences: ")

client = genai.Client(api_key=API_KEY)
ROOT = Path(__file__).parent


def discover_models():
    """Ask Google which models THIS api key can use, so we never call a model that doesn't exist."""
    text, tts, img = [], [], []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if acts and "generateContent" not in acts:
                continue
            low = name.lower()
            if "image" in low and "imagen" not in low:
                img.append(name)
            elif "tts" in low:
                tts.append(name)
            elif "flash" in low and not any(x in low for x in
                    ("image", "live", "audio", "native", "embedding", "robotics", "computer", "exp", "vision")):
                text.append(name)
    except Exception as e:
        print("Could not list models:", str(e)[:150], flush=True)
    order = lambda n: (("preview" in n), ("lite" in n), n)
    text = sorted(set(text), key=order)[:4]
    tts = sorted(set(tts), key=order)[:4]
    img = sorted(set(img), key=lambda n: (("pro" in n), ("preview" in n), n))[:3]
    return text, tts, img


_text, _tts, _img = discover_models()
TEXT_MODELS = list(dict.fromkeys([TEXT_MODEL] + _text)) or ["gemini-2.5-flash"]
TTS_MODELS = list(dict.fromkeys([m for m in [TTS_MODEL] if m] + _tts)) or ["gemini-2.5-flash-preview-tts"]
print("TEXT models to try:", TEXT_MODELS, flush=True)
IMG_MODELS = _img if os.getenv("GEMINI_IMAGES", "on").lower() != "off" else []
print("TTS models to try:", TTS_MODELS, flush=True)
print("IMAGE models to try:", IMG_MODELS, flush=True)


def log(*a):
    print(*a, flush=True)


TRANSIENT = ("503", "429", "500", "502", "504", "UNAVAILABLE", "overloaded", "RESOURCE_EXHAUSTED",
             "timed out", "Connection", "DEADLINE")


def retry(fn, tries=8, base=15):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if i == tries - 1 or not any(t in msg for t in TRANSIENT):
                raise
            wait = min(base * 2 ** i, 180) + random.randint(0, 10)
            log(f"  busy/error, retry {i+1}/{tries} in {wait}s: {msg[:120]}")
            time.sleep(wait)


def ask(prompt, json_out=False):
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if json_out else None
    last = None
    for model in TEXT_MODELS:
        try:
            r = retry(lambda m=model: client.models.generate_content(model=m, contents=prompt, config=cfg), tries=5)
            if r.text:
                return r.text.strip()
        except Exception as e:
            last = e
            log(f"  model {model} failed, trying next: {str(e)[:120]}")
    raise RuntimeError(f"All text models failed: {last}")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


# ---------- 1. SCRIPT ----------
def write_script(topic, d):
    of = d / "outline.json"
    if of.exists():
        outline = json.loads(of.read_text())
    else:
        log("Outline...")
        if STORY_STYLE == "story":
            outline_prompt = (f'Plan a calm, factual history sleep story titled "{topic}". Return a JSON list of exactly {SECTIONS} '
                              f'objects with keys "title" and "summary" (1 sentence). The story drifts slowly through one place and '
                              f'time period, moving from gentle arrival to deep rest.')
        else:
            outline_prompt = (f'Plan a calm, curious, factual history documentary for sleep titled "{topic}". Return a JSON list of exactly '
                              f'{SECTIONS} objects with keys "title" and "summary" (1 sentence). Each section covers a different aspect, '
                              f'example or era of the topic (if the title is a list like "15 things", cover the items across the sections). '
                              f'Start with a gentle introduction and end with a slow, quiet wrap-up.')
        outline = json.loads(ask(outline_prompt, json_out=True))
        of.write_text(json.dumps(outline))
    parts, prev_tail = [], ""
    for i, sec in enumerate(outline, 1):
        sf = d / f"sec_{i:02d}.txt"
        if sf.exists():
            text = sf.read_text()
        else:
            log(f"Section {i}/{len(outline)}: {sec['title']}")
            last = (i == len(outline))
            STYLE_RULES = ('- Second person, present tense ("You walk..."). Short, slow, gentle sentences. Rich sensory detail (light, sound, smell, warmth).'
                           if STORY_STYLE == "story" else
                           '- Calm, slow, warm documentary narration in third person, like a bedtime history lecture (you may say "imagine" now and then). '
                           'Rich concrete everyday detail: work, food, objects, smells, sounds, customs, with gentle curiosity and a little dry humour. Never rushed.')
            text = ask(f"""You are writing section {i} of {len(outline)} of a calm history documentary for sleep: "{topic}".
Full outline: {json.dumps(outline)}
This section: {sec['title']} - {sec['summary']}
Previous section ended with: "{prev_tail}"

Rules:
- About {WORDS_PER_SECTION} words, plain paragraphs only, no headings, no markdown, no lists.
{STYLE_RULES}
- Historically grounded. Only state details you are confident are accurate; when unsure, stay general. No invented statistics or dates dumps.
- Nothing graphic, violent or frightening. Dark topics are handled gently and matter-of-factly. No cliffhangers.
- Continue naturally from the previous section without recapping it.
{"- This is the opening section: begin with a short, warm welcome to the listener (settle in, dim the lights, breathe slowly), then gently introduce what tonight's video is about." if i == 1 else ""}
{"- This is the final section: slow down further and end with a soft, quiet fade into sleep." if last else ""}""")
            sf.write_text(text)
        parts.append(text)
        prev_tail = text[-250:]
    return outline, "\n\n".join(parts), parts


# ---------- 2. VOICE ----------
def chunk_text(text, limit=2500):
    chunks, cur = [], ""
    for p in text.split("\n\n"):
        if cur and len(cur) + len(p) > limit:
            chunks.append(cur.strip()); cur = ""
        cur += p + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def tts_gemini(chunks, d):
    files = []
    for i, c in enumerate(chunks):
        f = d / f"voice_{i:03d}.wav"
        if f.exists():
            files.append(f); continue
        log(f"Gemini TTS {i+1}/{len(chunks)}")
        r = None
        for model in TTS_MODELS:
            try:
                r = retry(lambda m=model: client.models.generate_content(
                    model=m, contents=NARRATION_STYLE + c,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE))))),
                    tries=4, base=30)
                break
            except Exception as e:
                log(f"  TTS model {model} failed, trying next: {str(e)[:120]}")
        if r is None:
            raise RuntimeError("all Gemini TTS models failed")
        pcm = r.candidates[0].content.parts[0].inline_data.data
        with wave.open(str(f), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
        files.append(f)
        time.sleep(4)
    return files


def tts_edge(chunks, d):
    import edge_tts
    files = []
    for i, c in enumerate(chunks):
        if (d / f"voice_{i:03d}.wav").exists():
            files.append(d / f"voice_{i:03d}.wav"); continue
        log(f"Edge-TTS {i+1}/{len(chunks)}")
        mp3 = d / f"voice_{i:03d}.mp3"
        asyncio.run(edge_tts.Communicate(c, EDGE_VOICE, rate="-15%", pitch="-5Hz").save(str(mp3)))
        wav = d / f"voice_{i:03d}.wav"
        run(["ffmpeg", "-y", "-i", str(mp3), "-ar", "24000", "-ac", "1", str(wav)])
        files.append(wav)
    return files


def make_voice(script, d):
    chunks = chunk_text(script)
    try:
        files = tts_gemini(chunks, d)
    except Exception as e:
        log("Gemini TTS failed, switching whole video to Edge-TTS:", e)
        for f in d.glob("voice_*"):
            f.unlink()
        files = tts_edge(chunks, d)
    lst = d / "voice_list.txt"
    lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    raw = d / "narration_raw.wav"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(raw)])
    out = d / "narration.wav"
    # even loudness + 6 seconds of quiet at the end
    run(["ffmpeg", "-y", "-i", str(raw), "-af", "loudnorm=I=-16:LRA=7:TP=-1.5,apad=pad_dur=6", str(out)])
    raw.unlink(missing_ok=True)
    return out


# ---------- helpers ----------
def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)]).strip())


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


# ---------- 3. TIMING: which word is spoken when ----------
def split_script(script):
    """Script -> sentences with word-index ranges [w0, w1) and a flag for 'last sentence of paragraph'."""
    sents, wi = [], 0
    for p in [p.strip() for p in script.split("\n\n") if p.strip()]:
        parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", p) if x.strip()]
        for k, sent in enumerate(parts):
            n = len(sent.split())
            sents.append({"text": sent, "w0": wi, "w1": wi + n, "para_end": k == len(parts) - 1})
            wi += n
    return sents, wi


def whisper_words(wav):
    from faster_whisper import WhisperModel
    log("Listening to the narration to get exact word timings (whisper)...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(wav), language="en", word_timestamps=True, beam_size=1,
                               condition_on_previous_text=False)
    out = []
    for sg in segs:
        for w in (sg.words or []):
            out.append((w.word.strip(), float(w.start), float(w.end)))
    return out


def align_words(script_words, wwords, total):
    """Match our script words to the words whisper heard, so every script word gets a start/end time."""
    sn = [norm(w) or f"_s{i}" for i, w in enumerate(script_words)]
    wn = [norm(w) or f"_w{j}" for j, (w, _, _) in enumerate(wwords)]
    n = len(sn)
    st, en = [None] * n, [None] * n
    for a, b, size in difflib.SequenceMatcher(None, sn, wn, autojunk=False).get_matching_blocks():
        for k in range(size):
            st[a + k], en[a + k] = wwords[b + k][1], wwords[b + k][2]
    known = [i for i in range(n) if st[i] is not None]
    if len(known) < 0.6 * n:
        raise RuntimeError(f"alignment too poor ({len(known)}/{n} words matched)")
    speech_end = max(total - 6.0, en[known[-1]])
    prev_t, i = 0.0, 0
    while i < n:
        if st[i] is not None:
            prev_t = en[i]; i += 1; continue
        j = i
        while j < n and st[j] is None:
            j += 1
        t1 = st[j] if j < n else min(speech_end, prev_t + 0.45 * (j - i))
        t1 = max(t1, prev_t)
        c = j - i
        for m in range(c):
            st[i + m] = prev_t + (t1 - prev_t) * m / c
            en[i + m] = prev_t + (t1 - prev_t) * (m + 1) / c
        i = j
    log(f"Timing aligned: {len(known)}/{n} words matched directly")
    return st, en


def word_times(script_words, narration, d):
    cache = d / "wordtimes.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if len(c["st"]) == len(script_words):
            return c["st"], c["en"]
    total = duration(narration)
    try:
        st, en = align_words(script_words, whisper_words(narration), total)
    except Exception as e:
        log("Whisper timing failed, using estimated timing instead:", str(e)[:200])
        speech, n = max(total - 6.0, 1.0), len(script_words)
        st = [speech * i / n for i in range(n)]
        en = [speech * (i + 1) / n for i in range(n)]
    cache.write_text(json.dumps({"st": st, "en": en}))
    return st, en


# ---------- 4. SUBTITLES ----------
def fmt_srt(t):
    t = max(t, 0.0)
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)) % 1000:03d}"


def make_cues(sents, st, en, max_chars=84):
    cues = []
    for s in sents:
        words = s["text"].split()
        piece, start_i = [], s["w0"]
        for k, w in enumerate(words):
            if piece and len(" ".join(piece + [w])) > max_chars:
                cues.append((st[start_i], en[start_i + len(piece) - 1], " ".join(piece)))
                piece, start_i = [], s["w0"] + k
            piece.append(w)
        if piece:
            cues.append((st[start_i], en[start_i + len(piece) - 1], " ".join(piece)))
    fixed = []
    for i, (a, b, txt) in enumerate(cues):
        b = max(b + 0.2, a + 0.6)
        if i + 1 < len(cues):
            b = min(b, cues[i + 1][0])
        fixed.append((a, max(b, a + 0.3), "\n".join(textwrap.wrap(txt, 44))))
    return fixed


def write_srt(cues, path):
    path.write_text("".join(f"{i}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1)))


SUB_STYLE = ("FontName=DejaVu Sans,FontSize=17,PrimaryColour=&H00F0F0F0&,OutlineColour=&H8C000000&,"
             "BackColour=&H8C000000&,BorderStyle=3,Outline=3,Shadow=0,MarginV=40,Alignment=2")


def subs_ok(d):
    try:
        (d / "t.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=1",
             "-vf", "subtitles=t.srt", "-frames:v", "1", "-f", "null", "-"], cwd=d)
        return True
    except Exception as e:
        log("Subtitle burn-in not available, skipping it:", str(e)[:200])
        return False


# ---------- 5. SCENES: one image per thing being talked about ----------
def build_scenes(sents):
    scenes, cur = [], []

    def close():
        nonlocal cur
        if cur:
            scenes.append({"text": " ".join(x["text"] for x in cur), "start": cur[0]["start"], "end": cur[-1]["end"]})
            cur = []

    for s in sents:
        cur.append(s)
        dur = s["end"] - cur[0]["start"]
        # change image at a sentence end; also at the end of a paragraph (topic change) if it already ran a while
        if dur >= IMAGE_SECONDS * 0.9 or (s["para_end"] and dur >= IMAGE_SECONDS * 0.55):
            close()
    close()
    if len(scenes) > 1 and scenes[-1]["end"] - scenes[-1]["start"] < 8:
        scenes[-2]["text"] += " " + scenes[-1]["text"]
        scenes[-2]["end"] = scenes[-1]["end"]
        scenes.pop()
    for i in range(len(scenes) - 1):
        scenes[i]["end"] = scenes[i + 1]["start"]
    scenes[0]["start"] = 0.0
    return scenes


STYLES = {
    "cinematic": ("dark moody cinematic photoreal film still, muted desaturated colours, dramatic low-key lighting, "
                  "warm lantern glow, fine film grain, figures seen from behind or in silhouette or far away, no visible faces, no text, no watermark"),
    "manuscript": ("medieval illuminated manuscript style illustration, flat gouache colours, small detailed figures, "
                   "ornate decorative border, aged parchment, no text, no watermark"),
    "engraving": ("black and white 19th century etching engraving illustration, fine crosshatching, aged paper, "
                  "detailed, no text, no watermark"),
}


def parse_items(txt):
    data = json.loads(txt)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out = []
    for x in data:
        if isinstance(x, dict):
            p = x.get("prompt") or next(iter(x.values()), "")
            st = str(x.get("style", "cinematic")).lower()
        else:
            p, st = str(x), "cinematic"
        out.append([str(p), st if st in STYLES else "cinematic"])
    return out


def scene_prompts(topic, scenes, d):
    pf = d / "prompts.json"
    prompts = None
    if pf.exists():
        cached = json.loads(pf.read_text())
        if len(cached) == len(scenes) and cached and isinstance(cached[0], list):
            prompts = cached
    if prompts is None:
        prompts, batch = [], 15
        order = ["cinematic", "manuscript", "cinematic", "engraving", "manuscript"]
        for i0 in range(0, len(scenes), batch):
            part = scenes[i0:i0 + batch]
            listing = "\n".join(f"{k+1}. {s['text'][:500]}" for k, s in enumerate(part))
            txt = ask(f"""Story: "{topic}" (a calm history documentary for sleep).
Below are numbered passages from the narration. For EACH passage write an image prompt (max 30 words) showing exactly
what that passage describes: the place, people, objects, time of day and light. Calm, no text or letters, no modern
objects, no gore. NEVER describe a close-up of a face: people are small in the frame, seen from behind, or in silhouette.
Also choose the best "style" for it:
- "cinematic": places, dark streets, interiors, objects, hands, candles, fire, people only far away or from behind (about 40%)
- "manuscript": any passage with people doing things: customs, trade, law, daily life, small stylised figures (about 40%)
- "engraving": tools, animals, buildings, objects, old documents (about 20%)
Return a JSON list of exactly {len(part)} objects like {{"prompt": "...", "style": "cinematic"}}, in order.

{listing}""", json_out=True)
            try:
                got = parse_items(txt)
            except Exception:
                got = []
            fb = [[s["text"][:120], order[(i0 + k) % len(order)]] for k, s in enumerate(part)]
            prompts += (got + fb[len(got):])[:len(part)]
        pf.write_text(json.dumps(prompts))
    for sc, (_, st) in zip(scenes, prompts):
        sc["style"] = st
    return prompts


def placeholder(path):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x0b1530:s={OUT_W}x{OUT_H}", "-frames:v", "1", str(path)])


def make_image(prompt, path, seed, style_text=None):
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(f"{prompt}, {style_text or STYLE}") +
           f"?width={OUT_W}&height={OUT_H}&seed={seed}&nologo=true&model=flux")
    for t in range(4):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 5000:
                path.write_bytes(r.content)
                return True
        except Exception as e:
            log("  image error:", str(e)[:100])
        time.sleep(15 * (t + 1))
    return False


IMG_STATE = {"ok": True, "fails": 0}


def gemini_image(prompt, path):
    """Try this key's Gemini image models (best quality). Returns False if no quota, so we fall back to Pollinations."""
    if not IMG_MODELS or not IMG_STATE["ok"]:
        return False
    for model in IMG_MODELS:
        try:
            kw = dict(response_modalities=["TEXT", "IMAGE"])
            ic = getattr(types, "ImageConfig", None)
            if ic:
                kw["image_config"] = ic(aspect_ratio="16:9")
            r = client.models.generate_content(model=model, contents=prompt,
                                               config=types.GenerateContentConfig(**kw))
            for part in r.candidates[0].content.parts:
                data = getattr(getattr(part, "inline_data", None), "data", None)
                if data:
                    path.write_bytes(data)
                    IMG_STATE["fails"] = 0
                    return True
            raise RuntimeError("no image in reply")
        except Exception as e:
            log(f"  Gemini image ({model}) failed: {str(e)[:140]}")
    IMG_STATE["fails"] += 1
    if IMG_STATE["fails"] >= 2:
        IMG_STATE["ok"] = False
        log("Gemini images not available right now (quota) -> using Pollinations for the rest")
    return False


def find_img(d, i):
    for ext in (".png", ".jpg"):
        p = d / f"img_{i:03d}{ext}"
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def make_images(topic, scenes, d):
    prompts = scene_prompts(topic, scenes, d)
    imgs, last_ok = [], None
    for i, (p, st) in enumerate(prompts):
        have = find_img(d, i)
        if have:
            imgs.append(have); last_ok = have; continue
        log(f"Image {i+1}/{len(prompts)}")
        png, jpg = d / f"img_{i:03d}.png", d / f"img_{i:03d}.jpg"
        if gemini_image(f"{p}. {STYLES[st]}. 16:9 landscape.", png):
            imgs.append(png); last_ok = png
            time.sleep(4)
            continue
        if make_image(p, jpg, seed=random.randint(1, 10**6), style_text=STYLES[st]):
            last_ok = jpg
        elif last_ok:
            jpg = d / f"img_{i:03d}{last_ok.suffix}"
            shutil.copy(last_ok, jpg)
        else:
            placeholder(jpg)
        imgs.append(jpg)
        time.sleep(6)   # be gentle with the free service
    return imgs


# ---------- 6. MUSIC (free, made by this script, no copyright issue) ----------
def make_ambient(path, seconds=60):
    sr = 44100
    t = np.arange(sr * seconds) / sr
    sig = np.zeros_like(t)
    # whole-number frequencies + LFO periods that divide 60 s => the loop is seamless
    for f, amp, period, ph in [(55, .5, 60, 0), (110, .6, 30, 1), (165, .4, 20, 2),
                               (220, .35, 15, 3), (277, .2, 12, 4), (330, .25, 20, 5)]:
        lfo = 0.55 + 0.45 * np.sin(2 * np.pi * t / period + ph)
        for det in (0.0, 0.1, -0.1):
            sig += amp * lfo * np.sin(2 * np.pi * (f + det) * t + ph)
    sig = sig / np.abs(sig).max() * 0.6
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())


def pick_music(d):
    if os.getenv("MUSIC", "on").lower() == "off":
        return None
    mine = next(iter((ROOT / "assets").glob("music.*")), None)
    if mine:
        return mine
    amb = d / "ambient.wav"
    if not amb.exists():
        make_ambient(amb)
    return amb


# ---------- 7. VIDEO (smooth zoom drawn frame by frame, crossfades, subtitles, music) ----------
def load_src(path):
    im = cv2.imread(str(path))
    if im is None:
        im = np.full((OUT_H, OUT_W, 3), (48, 21, 11), np.uint8)
    h, w = im.shape[:2]
    if Path(path).suffix.lower() == ".jpg":   # Pollinations images: cut off the watermark corner
        im = im[: max(int(h * (1 - CROP_BOTTOM)), 2), : max(int(w * (1 - CROP_RIGHT)), 2)]
    if im.shape[1] < OUT_W:
        factor = OUT_W / im.shape[1]
        im = cv2.resize(im, (OUT_W, int(im.shape[0] * factor)), interpolation=cv2.INTER_LANCZOS4)
        if factor > 1.25:
            im = cv2.addWeighted(im, 1.6, cv2.GaussianBlur(im, (0, 0), 2.2), -0.6, 0)
    return np.ascontiguousarray(im)


SHOT_PRESETS = [
    dict(z0=1.04, z1=1.14, fx=.5, fy=.5, dx=.0, dy=.0, kz=.16, lat=.6),      # wide "drone" push-in with parallax
    dict(z0=1.30, z1=1.40, fx=.15, fy=.45, dx=.25, dy=.0, kz=.10, lat=.3),   # closer, left side
    dict(z0=1.30, z1=1.40, fx=.85, fy=.5, dx=-.25, dy=.0, kz=.10, lat=.3),   # closer, right side
    dict(z0=1.45, z1=1.35, fx=.5, fy=.35, dx=.0, dy=.12, kz=.06, lat=.2),    # detail, upper centre
]


def draw_view(src, u, p):
    """A smoothly moving 16:9 window over the picture; u = 0..1 progress. Sub-pixel exact, so no jitter."""
    h, w = src.shape[:2]
    z = p["z0"] + (p["z1"] - p["z0"]) * u
    ww = min(w, h * OUT_W / OUT_H) / z
    wh = ww * OUT_H / OUT_W
    x0 = (w - ww) * min(max(p["fx"] + p["dx"] * (u - 0.5), 0.0), 1.0)
    y0 = (h - wh) * min(max(p["fy"] + p["dy"] * (u - 0.5), 0.0), 1.0)
    s = OUT_W / ww
    m = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(src, m, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def list_broll():
    """Short video clips you made yourself (Grok, fal.ai, Flow...). Put them in assets/broll/ ."""
    out = []
    for p in sorted((ROOT / "assets" / "broll").glob("*")):
        if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"):
            cap = cv2.VideoCapture(str(p))
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 24.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if n and n / fps > 1.0:
                out.append({"path": p, "dur": n / fps})
    return out


class ClipReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.pos, self.last = 0, None

    def frame(self, lt):
        idx = int(max(lt, 0.0) * self.fps)
        while self.pos <= idx:
            ok, f = self.cap.read()
            if not ok:
                break
            self.last, self.pos = f, self.pos + 1
        if self.last is None:
            return np.zeros((OUT_H, OUT_W, 3), np.uint8)
        h, w = self.last.shape[:2]
        sc = max(OUT_W / w, OUT_H / h)
        im = cv2.resize(self.last, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (im.shape[0] - OUT_H) // 2, (im.shape[1] - OUT_W) // 2
        return np.ascontiguousarray(im[y0:y0 + OUT_H, x0:x0 + OUT_W])

    def close(self):
        self.cap.release()


_DEPTH = {"pipe": None, "failed": False}
_GRID = {}


def get_depth(src):
    """Real depth map from a small free model (if installed); None if not available."""
    if _DEPTH["failed"]:
        return None
    try:
        if _DEPTH["pipe"] is None:
            from transformers import pipeline
            log("Loading depth model for 3D drone shots...")
            _DEPTH["pipe"] = pipeline("depth-estimation",
                                      model=os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"), device=-1)
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(cv2.resize(src, (768, int(768 * src.shape[0] / src.shape[1]))), cv2.COLOR_BGR2RGB))
        dep = _DEPTH["pipe"](pil)["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
        dep = cv2.resize(dep, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_CUBIC)
        dep = (dep - dep.min()) / (dep.max() - dep.min() + 1e-6)      # 1 = near, 0 = far
        return cv2.GaussianBlur(dep, (0, 0), 4)
    except Exception as e:
        _DEPTH["failed"] = True
        log("Depth model not available, using a simple fake depth instead:", str(e)[:120])
        return None


def fake_depth(src):
    """Ground is near (bottom), sky/horizon is far (top): gives a believable flying-forward feel."""
    h, w = src.shape[:2]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None] ** 1.3
    return np.ascontiguousarray(np.repeat(y, w, axis=1))


def draw_view_3d(src, depth, u, p):
    """Camera flies forward with real parallax: near things grow faster than far things.
    Maps are computed at 1/4 size and enlarged (they are smooth), then ONE remap draws the frame."""
    h, w = src.shape[:2]
    z = p["z0"] + (p["z1"] - p["z0"]) * u
    ww = min(w, h * OUT_W / OUT_H) / z
    wh = ww * OUT_H / OUT_W
    x0 = (w - ww) * min(max(p["fx"] + p["dx"] * (u - 0.5), 0.0), 1.0)
    y0 = (h - wh) * min(max(p["fy"] + p["dy"] * (u - 0.5), 0.0), 1.0)
    sc = OUT_W / ww
    lw, lh = OUT_W // 4, OUT_H // 4
    if "lx" not in _GRID:
        _GRID["lx"], _GRID["ly"] = np.meshgrid((np.arange(lw, dtype=np.float32) * 4 + 1.5),
                                               (np.arange(lh, dtype=np.float32) * 4 + 1.5))
    m_low = np.float32([[sc / 4, 0, -x0 * sc / 4], [0, sc / 4, -y0 * sc / 4]])
    dwin = cv2.warpAffine(depth, m_low, (lw, lh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cx, cy = (OUT_W - 1) / 2, (OUT_H - 1) / 2
    f = 1.0 + p["kz"] * u * dwin
    ox = cx + (_GRID["lx"] - cx) / f + p["lat"] * (u - 0.5) * 2 * (dwin - 0.5) * OUT_W * 0.012
    oy = cy + (_GRID["ly"] - cy) / f
    sx = cv2.resize(x0 + ox / sc, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    sy = cv2.resize(y0 + oy / sc, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(src, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_VIG = {}


def vignette(frame, strength=0.35):
    if "m" not in _VIG:
        y, x = np.mgrid[:OUT_H, :OUT_W].astype(np.float32)
        r = np.sqrt(((x - OUT_W / 2) / (OUT_W / 2)) ** 2 + ((y - OUT_H / 2) / (OUT_H / 2)) ** 2)
        m = np.clip(1.0 - strength * np.clip(r - 0.55, 0, 1) ** 1.5, 0.55, 1.0)
        _VIG["m"] = np.ascontiguousarray(np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2))
    return cv2.multiply(frame, _VIG["m"], scale=1 / 255.0)


def build_shots(scenes, total, clips=()):
    """Each picture is cut into several short shots (wide, close-left, close-right...), like the reference channel.
    If you added video clips, one of them plays at the start of every BROLL_EVERY-th picture."""
    shots, rnd, pool = [], random.Random(1234), []
    for k, sc in enumerate(scenes):
        end = total if k == len(scenes) - 1 else scenes[k + 1]["start"]
        length = end - sc["start"]
        n = max(1, round(length / SHOT_SECONDS))
        clip = sc.get("clip")
        if clip is None and clips and k % BROLL_EVERY == 0:
            if not pool:
                pool = list(clips)
                rnd.shuffle(pool)
            clip = pool.pop()
        for j in range(n):
            a, b = sc["start"] + length * j / n, sc["start"] + length * (j + 1) / n
            fi = FADE if j == 0 else CUT_FADE
            if j == 0 and clip:
                vd = min(clip["dur"], b - a)
                if b - (a + vd) < 1.5:
                    vd = b - a
                shots.append({"kind": "video", "path": clip["path"], "img": k, "style": "cinematic", "p": None,
                              "start": a, "end": a + vd, "fi": fi})
                if b - (a + vd) > 0.1:
                    shots.append({"img": k, "style": sc.get("style", "cinematic"), "p": dict(SHOT_PRESETS[0]),
                                  "start": a + vd, "end": b, "fi": CUT_FADE})
                continue
            p = dict(SHOT_PRESETS[0] if j == 0 else SHOT_PRESETS[1 + (k + j) % 3])
            if rnd.random() < 0.5:
                p["z0"], p["z1"] = p["z1"], p["z0"]
            shots.append({"img": k, "style": sc.get("style", "cinematic"), "p": p, "start": a, "end": b, "fi": fi})
    shots[0]["fi"] = 0.0
    return shots


class Embers:
    """Little glowing sparks drifting upwards, drawn on every frame (the 'fire flares' look)."""
    def __init__(self, count=70, seed=5):
        r = np.random.default_rng(seed)
        self.n = count
        self.x0 = r.uniform(0, OUT_W, count)
        self.y0 = r.uniform(0, OUT_H + 60, count)
        self.vy = r.uniform(22, 80, count)
        self.drift = r.uniform(-10, 16, count)
        self.amp = r.uniform(8, 40, count)
        self.sw = r.uniform(0.2, 0.7, count)
        self.ph = r.uniform(0, 6.28, count)
        self.flick = r.uniform(1.5, 4.0, count)
        self.rad = r.choice([1.2, 1.8, 2.6, 3.8, 6.5, 9.0], count, p=[.3, .3, .17, .1, .08, .05])
        self.cache = {}

    def sprite(self, rad):
        if rad not in self.cache:
            size = int(rad * 7) | 1
            yy, xx = np.mgrid[:size, :size] - size // 2
            g = np.exp(-(xx ** 2 + yy ** 2) / (2 * rad ** 2)).astype(np.float32)
            core = g ** 4
            bgr = np.stack([40 * g + 150 * core, 140 * g + 190 * core, 255 * g + 200 * core], -1)
            self.cache[rad] = (bgr * (1.0 if rad <= 3.8 else 0.35)).astype(np.float32)
        return self.cache[rad]

    def draw(self, frame, t, gain=1.0):
        if gain <= 0.02:
            return
        for i in range(self.n):
            y = (self.y0[i] - self.vy[i] * t) % (OUT_H + 60) - 30
            x = (self.x0[i] + self.drift[i] * t + self.amp[i] * np.sin(self.sw[i] * t + self.ph[i])) % OUT_W
            fl = 0.55 + 0.45 * np.sin(self.flick[i] * t + self.ph[i] * 2)
            edge = min(1.0, (y + 30) / 140, (OUT_H + 30 - y) / 140)
            inten = fl * gain * max(edge, 0.0)
            spr = self.sprite(float(self.rad[i]))
            hh, ww = spr.shape[:2]
            xi, yi = int(x) - ww // 2, int(y) - hh // 2
            x1, y1, x2, y2 = max(xi, 0), max(yi, 0), min(xi + ww, OUT_W), min(yi + hh, OUT_H)
            if x2 <= x1 or y2 <= y1:
                continue
            reg = frame[y1:y2, x1:x2].astype(np.int16) + (spr[y1 - yi:y2 - yi, x1 - xi:x2 - xi] * inten).astype(np.int16)
            frame[y1:y2, x1:x2] = np.clip(reg, 0, 255).astype(np.uint8)


EMBER_GAIN = {"cinematic": 0.9, "engraving": 0.35, "manuscript": 0.25}


def load_logo():
    p = next(iter((ROOT / "assets").glob("logo.*")), None)
    if not p:
        return None
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    h = 110
    im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
    if im.shape[2] == 4:
        return im[..., :3].astype(np.float32), (im[..., 3:4].astype(np.float32) / 255.0)
    return im.astype(np.float32), np.full(im.shape[:2] + (1,), 0.9, np.float32)


def put_logo(frame, logo):
    bgr, a = logo
    h, w = bgr.shape[:2]
    y1, x1 = OUT_H - h - 36, 36
    reg = frame[y1:y1 + h, x1:x1 + w].astype(np.float32)
    frame[y1:y1 + h, x1:x1 + w] = (reg * (1 - a) + bgr * a).astype(np.uint8)


def render_video(scenes, imgs, narration, out, d, use_subs, use_pool=True):
    total = duration(narration)
    clips = list_broll() if use_pool else []
    shots = build_shots(scenes, total, clips)
    n = len(shots)
    log(f"{len(scenes)} pictures + {len(clips)} video clips available -> {n} shots (about {total / n:.1f}s each)")
    for i, sh in enumerate(shots):
        sh["a"] = 0.0 if i == 0 else sh["start"] - sh["fi"] / 2
        sh["b"] = total if i == n - 1 else shots[i + 1]["start"] + shots[i + 1]["fi"] / 2
    cache = {}
    embers = Embers() if EMBERS else None
    logo = load_logo()

    def fr(k, t):
        sh = shots[k]
        if sh.get("kind") == "video":
            if "reader" not in sh:
                sh["reader"] = ClipReader(sh["path"])
            if k >= 2 and "reader" in shots[k - 2]:
                shots[k - 2].pop("reader").close()
            return sh["reader"].frame(t - sh["start"])
        key = sh["img"]
        if key not in cache:
            for old in [x for x in cache if x < key - 1]:
                del cache[old]
            cache[key] = {"src": load_src(imgs[key]), "depth": None}
        ent = cache[key]
        u = min(max((t - sh["a"]) / max(sh["b"] - sh["a"], 1e-6), 0.0), 1.0)
        if sh["style"] == "cinematic" and sh["p"].get("kz", 0) > 0 and THREE_D:
            if ent["depth"] is None:
                ent["depth"] = get_depth(ent["src"])
                if ent["depth"] is None:
                    ent["depth"] = fake_depth(ent["src"])
            return draw_view_3d(ent["src"], ent["depth"], u, sh["p"])
        return draw_view(ent["src"], u, sh["p"])

    music = pick_music(d)
    vf = "[0:v]" + (f"subtitles=subs.srt:original_size={OUT_W}x{OUT_H}:force_style='{SUB_STYLE}'," if use_subs else "") \
         + "scale=out_color_matrix=bt709,format=yuv420p,noise=alls=5:allf=t[v]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
           "-framerate", str(FPS), "-i", "-", "-i", str(narration.resolve())]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        af = ("[2:a]lowpass=f=1800,aecho=0.8:0.6:700|1100:0.25|0.15,volume=0.12[m];"
              "[1:a][m]amix=inputs=2:duration=first:normalize=0[a]")
        cmd += ["-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-filter_complex", vf, "-map", "[v]", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-maxrate", "3500k", "-bufsize", "7M",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(Path(out).resolve())]
    logf = open(d / "ffmpeg.log", "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=logf, cwd=d)
    nframes, base, t_start = int(total * FPS), 0, time.time()
    try:
        for f in range(nframes):
            t = f / FPS
            while base + 1 < n and t >= shots[base + 1]["start"]:
                base += 1
            nxt_fi = shots[base + 1]["fi"] if base + 1 < n else 0.0
            if base + 1 < n and nxt_fi > 0 and t >= shots[base + 1]["start"] - nxt_fi / 2:
                a = (t - (shots[base + 1]["start"] - nxt_fi / 2)) / nxt_fi
                frame = cv2.addWeighted(fr(base, t), 1 - a, fr(base + 1, t), a, 0)
                style_gain = (1 - a) * EMBER_GAIN[shots[base]["style"]] + a * EMBER_GAIN[shots[base + 1]["style"]]
            elif base >= 1 and shots[base]["fi"] > 0 and t < shots[base]["start"] + shots[base]["fi"] / 2:
                fi = shots[base]["fi"]
                a = (t - (shots[base]["start"] - fi / 2)) / fi
                frame = cv2.addWeighted(fr(base - 1, t), 1 - a, fr(base, t), a, 0)
                style_gain = (1 - a) * EMBER_GAIN[shots[base - 1]["style"]] + a * EMBER_GAIN[shots[base]["style"]]
            else:
                frame = fr(base, t)
                style_gain = EMBER_GAIN[shots[base]["style"]]
            if style_gain > 0.5:      # photo-style scenes: darker edges and a slight firelight flicker
                frame = vignette(frame)
                flick = 1.0 + 0.025 * math.sin(t * 9.0) * math.sin(t * 2.3 + 1.0)
                frame = cv2.convertScaleAbs(frame, alpha=flick)
            if embers:
                embers.draw(frame, t, style_gain)
            if logo is not None:
                put_logo(frame, logo)
            g = min(1.0, t / 2.0, (total - t) / 4.0)
            if g < 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=max(g, 0.0))
            proc.stdin.write(frame.tobytes())
            if f % (FPS * 120) == 0:
                log(f"Rendering {100 * f // nframes}%  ({int(time.time() - t_start)}s)")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait()
    logf.close()
    if rc != 0:
        raise RuntimeError("ffmpeg failed: " + (d / "ffmpeg.log").read_text()[-800:])


# ---------- 8. THUMBNAILS ----------
def _font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_thumbnail(img_path, topic, out, variant=0, minutes=None):
    from PIL import Image, ImageDraw, ImageEnhance, ImageOps
    W, H = 1280, 720
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if Path(img_path).suffix.lower() == ".jpg":
            im = im.crop((0, 0, int(w * (1 - CROP_RIGHT)), int(h * (1 - CROP_BOTTOM))))
        im = ImageOps.fit(im, (W, H), Image.LANCZOS)
        im = ImageEnhance.Brightness(ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(1.18)).enhance(1.2)).enhance(1.05)
        arr = np.asarray(im).astype(np.float32)
        xs, ys = np.linspace(0, 1, W)[None, :], np.linspace(0, 1, H)[:, None]
        if variant == 0:
            side = np.clip(1.25 - xs * 1.9, 0, 1) * np.ones((H, 1))
        elif variant == 1:
            side = np.clip(xs * 1.9 - 0.65, 0, 1) * np.ones((H, 1))
        else:
            side = np.clip(ys * 1.9 - 0.5, 0, 1) * np.ones((1, W))
        vig = np.clip(np.sqrt((xs - .5) ** 2 + (ys - .5) ** 2) * 1.2, 0, 1) ** 2 * 0.5
        dark = np.clip(side * 0.78 + vig, 0, 0.9)
        im = Image.fromarray((arr * (1 - dark[..., None])).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        box_w = 1100 if variant == 2 else 640
        lines, size = textwrap.wrap(topic.upper(), 14), 130
        while size > 50:
            f = _font(size)
            widths = [dr.textlength(l, font=f) for l in lines]
            if max(widths) <= box_w and len(lines) * size * 1.12 <= 400:
                break
            size -= 6
        f = _font(size)
        line_h = int(size * 1.12)
        th = line_h * len(lines)
        y = (H - th) // 2 if variant < 2 else H - 70 - th
        for k, l in enumerate(lines):
            lw = dr.textlength(l, font=f)
            x = 60 if variant == 0 else (W - 60 - lw if variant == 1 else (W - lw) / 2)
            dr.text((x, y + k * line_h), l, font=f, fill=(255, 214, 120) if k == len(lines) - 1 else (255, 244, 225),
                    stroke_width=7, stroke_fill=(10, 8, 20))
        label = f"{minutes} MIN SLEEP STORY" if minutes else "SLEEP STORY"
        bf = _font(36)
        bw = dr.textlength(label, font=bf)
        bx = 60 if variant == 0 else (W - 60 - bw - 40 if variant == 1 else (W - bw - 40) / 2)
        dr.rounded_rectangle((bx, 48, bx + bw + 40, 48 + 62), radius=30, fill=(15, 20, 45), outline=(255, 214, 120), width=3)
        dr.text((bx + 20, 56), label, font=bf, fill=(255, 255, 255))
        im.save(out, quality=92)
    except Exception as e:
        log("Thumbnail design failed, saving plain image:", str(e)[:150])
        shutil.copy(img_path, out)


def gemini_thumbnail(topic, minutes, variant, out):
    looks = ["a wide, dramatic establishing view of the place with its most famous landmark",
             "a warm street-level view with glowing lanterns and a few people walking",
             "a cozy peaceful view through a window with a candle and the city beyond"]
    prompt = (f'Create a premium 16:9 YouTube thumbnail for a calm history sleep story. Scene: {looks[variant]} for "{topic}", '
              f'at night with a glowing full moon, deep navy sky, warm golden lantern light, rich cinematic detail. '
              f'Add elegant large golden serif title text "{topic}" (the last words bigger), and a small top badge with a '
              f'crescent moon icon and the words "{minutes} MIN SLEEP STORY". Text must be spelled exactly, sharp and readable. '
              f'High contrast, polished, no watermark.')
    return gemini_image(prompt, out)


# ---------- 9. METADATA ----------
def fmt_chapter(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def make_metadata(topic, outline, chapters):
    m = json.loads(ask(f"""YouTube metadata for a calm history sleep story about "{topic}" (US audience).
Return JSON with keys: "title" (max 70 chars, like "{topic} | Relaxing History for Sleep"),
"description" (3-4 calm sentences only, no chapter list),
"tags" (list of 12 short tags).""", json_out=True))
    if chapters:
        m["description"] += "\n\nChapters:\n" + "\n".join(chapters)
    m["description"] += "\n\nNarration and visuals in this video are AI-generated. Educational, relaxing history storytelling."
    m["thumbnail_prompt"] = (f'YouTube thumbnail, 16:9, cinematic and dramatic, moonlit ancient scene for "{topic}", warm glowing '
                             f'lights against deep blue night, big bold readable title text "{topic.upper()}", small badge "SLEEP STORY", '
                             f'high contrast, painterly, no watermark.')
    return m


# ---------- MAIN ----------
def pick_topic():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    done = set((ROOT / "done.txt").read_text().splitlines()) if (ROOT / "done.txt").exists() else set()
    for line in (ROOT / "topics.txt").read_text().splitlines():
        if line.strip() and line.strip() not in done:
            return line.strip()
    sys.exit("No topics left in topics.txt")


MOTION = {
    "cinematic": "Slow cinematic camera push-in. Candle or firelight flickers softly, fog and dust drift through the air, tiny embers float upwards. Subtle natural movement only, no cuts, no text, no new objects.",
    "manuscript": "Gently animate this medieval painting: small figures move slightly, smoke rises, flames flicker, slow camera drift. Keep the painted style exactly, no cuts, no text.",
    "engraving": "Gently animate this engraving: slight movement of figures and animals, smoke or steam rising, slow camera drift. Keep the black and white etched style exactly, no cuts, no text.",
}


def full_image_prompt(sc):
    return f"{sc['prompt']}. {STYLES[sc['style']]}. Wide 16:9 landscape image, cinematic composition, high detail."


def write_plan(topic, d, outline, parts, script, narration, st, sents, scenes, prompts, total):
    plan = d / "plan"
    plan.mkdir(exist_ok=True)
    for sc, (p, stl) in zip(scenes, prompts):
        sc["prompt"], sc["style"] = p, stl
    chapters, off = [], 0
    for i, sec in enumerate(outline[:len(parts)]):
        chapters.append(f"{fmt_chapter(0 if i == 0 else st[min(off, len(st) - 1)])} {sec['title']}")
        off += len(parts[i].split())
    meta = make_metadata(topic, outline, chapters if len(chapters) >= 3 else [])
    (plan / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    shutil.copy(d / "subs.srt", plan / "subs.srt")
    (plan / "script.txt").write_text(script)
    run(["ffmpeg", "-y", "-i", str(narration), "-ac", "1", "-b:a", "96k", str(plan / "narration.mp3")])
    minutes = int(total // 60)
    (plan / "plan.json").write_text(json.dumps({
        "topic": topic, "minutes": minutes,
        "scenes": [{"start": s["start"], "end": s["end"], "text": s["text"], "style": s["style"], "prompt": s["prompt"]}
                   for s in scenes]}, ensure_ascii=False))
    stars = [i + 1 for i in range(len(scenes)) if i % BROLL_EVERY == 0]
    out = [f"# {topic}", f"{minutes} min video, {len(scenes)} scenes.\n",
           "HOW TO USE",
           "1. For each scene below, paste the IMAGE PROMPT into Gemini / ChatGPT / Grok / Flow and save the picture.",
           "2. Name the file with the scene number: 001.png, 002.png, 003.png ...",
           "3. For scenes marked VIDEO, upload your picture to an image-to-video tool, paste the VIDEO PROMPT, save the clip as 001.mp4 (same number).",
           "4. Upload everything to the 'media' folder of your GitHub repo. Scenes you skip are made automatically by the script.",
           f"5. Thumbnail: make it yourself and name it thumbnail.png (prompt at the bottom).",
           f"\nMINIMUM: the VIDEO scenes ({len(stars)} of them): {', '.join(f'{n:03d}' for n in stars[:40])}{' ...' if len(stars) > 40 else ''}\n"]
    for i, sc in enumerate(scenes):
        n = i + 1
        out.append(f"## {n:03d}  {fmt_chapter(sc['start'])}-{fmt_chapter(sc['end'])}  [{sc['style']}]" + ("  VIDEO" if i % BROLL_EVERY == 0 else ""))
        out.append(f"Narration: {sc['text'][:220]}{'...' if len(sc['text']) > 220 else ''}")
        out.append(f"IMAGE PROMPT: {full_image_prompt(sc)}")
        if i % BROLL_EVERY == 0:
            out.append(f"VIDEO PROMPT: {MOTION[sc['style']]} (5-8 seconds, 16:9)")
        out.append("")
    out.append("THUMBNAIL PROMPT: " + meta["thumbnail_prompt"])
    out.append("\nTITLE: " + meta["title"])
    (plan / "PLAN.txt").write_text("\n".join(out))
    for f in list(d.glob("voice_*")) + list(d.glob("sec_*.txt")) + [d / "narration.wav", d / "voice_list.txt", d / "outline.json",
                                                                      d / "prompts.json", d / "wordtimes.json"]:
        f.unlink(missing_ok=True)
    log(f"PLAN READY -> {plan / 'PLAN.txt'}  ({len(scenes)} scenes)")


def number_of(p):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None


def collect_media():
    imgs, vids, thumbs = {}, {}, []
    md = ROOT / "media"
    for p in (sorted(md.glob("*")) if md.exists() else []):
        ext = p.suffix.lower()
        if p.stem.lower().startswith("thumb"):
            thumbs.append(p); continue
        n = number_of(p)
        if n is None:
            continue
        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            imgs[n] = p
        elif ext in (".mp4", ".mov", ".webm", ".mkv"):
            vids[n] = p
    return imgs, vids, thumbs


def assemble_main():
    pj = next(iter(Path(os.getenv("PLAN_DIR", "plan_in")).rglob("plan.json")), None)
    if not pj:
        sys.exit("plan.json not found - check the plan_run_id you typed")
    plan_dir = pj.parent
    data = json.loads(pj.read_text())
    topic, scenes = data["topic"], data["scenes"]
    d = ROOT / "output" / slugify(topic)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(plan_dir / "subs.srt", d / "subs.srt")
    (d / "prompts.json").write_text(json.dumps([[sc["prompt"], sc["style"]] for sc in scenes]))
    imgs_u, vids_u, thumbs = collect_media()
    log(f"TOPIC: {topic} | {len(scenes)} scenes | your media: {len(imgs_u)} images, {len(vids_u)} clips, {len(thumbs)} thumbnails")
    for i, sc in enumerate(scenes):
        n, png = i + 1, d / f"img_{i:03d}.png"
        if n in imgs_u:
            im = cv2.imread(str(imgs_u[n]))
            if im is not None:
                cv2.imwrite(str(png), im)
        if n in vids_u:
            cap = cv2.VideoCapture(str(vids_u[n]))
            fps, frames = cap.get(cv2.CAP_PROP_FPS) or 24.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not png.exists():
                ok, f = cap.read()
                if ok:
                    cv2.imwrite(str(png), f)
            cap.release()
            if frames and frames / fps > 1.0:
                sc["clip"] = {"path": vids_u[n], "dur": frames / fps}
    narration = plan_dir / "narration.mp3"
    imgs = make_images(topic, scenes, d)          # only fills the scenes you did not upload
    render_video(scenes, imgs, narration, d / "video.mp4", d, BURN_SUBS and subs_ok(d), use_pool=False)
    shutil.copy(plan_dir / "metadata.json", d / "metadata.json")
    shutil.copy(plan_dir / "script.txt", d / "script.txt")
    if thumbs:
        for k, t in enumerate(thumbs):
            shutil.copy(t, d / f"thumbnail_yours_{k + 1}{t.suffix.lower()}")
    else:
        for k, pos in enumerate((0.15, 0.45, 0.75)):
            make_thumbnail(imgs[min(int(len(imgs) * pos), len(imgs) - 1)], topic, d / f"thumbnail_{'abc'[k]}.jpg",
                           variant=k, minutes=data["minutes"])
    for f in list(d.glob("voice_*")) + [d / "prompts.json", d / "t.srt"]:
        f.unlink(missing_ok=True)
    log("DONE ->", d / "video.mp4")


def main():
    if MODE == "assemble":
        return assemble_main()
    topic = pick_topic()
    log("TOPIC:", topic, "| mode:", MODE)
    d = ROOT / "output" / slugify(topic)
    d.mkdir(parents=True, exist_ok=True)

    outline, script, parts = write_script(topic, d)
    (d / "script.txt").write_text(script)
    narration = make_voice(script, d)

    sents, nwords = split_script(script)
    script_words = [w for s in sents for w in s["text"].split()]
    st, en = word_times(script_words, narration, d)
    for s in sents:
        s["start"] = st[s["w0"]]
        s["end"] = max(en[s["w1"] - 1], s["start"] + 0.3)
    write_srt(make_cues(sents, st, en), d / "subs.srt")
    total = duration(narration)
    scenes = build_scenes(sents)
    log(f"{len(scenes)} scenes (one image per scene)")

    if MODE == "plan":
        prompts = scene_prompts(topic, scenes, d)
        write_plan(topic, d, outline, parts, script, narration, st, sents, scenes, prompts, total)
        with open(ROOT / "done.txt", "a") as f:
            f.write(topic + "\n")
        return

    thumb_done = [gemini_thumbnail(topic, int(total // 60), k, d / f"thumbnail_{'abc'[k]}.png") for k in range(3)]
    imgs = make_images(topic, scenes, d)

    render_video(scenes, imgs, narration, d / "video.mp4", d, BURN_SUBS and subs_ok(d))

    chapters, off = [], 0
    for i, sec in enumerate(outline[:len(parts)]):
        chapters.append(f"{fmt_chapter(0 if i == 0 else st[min(off, len(st) - 1)])} {sec['title']}")
        off += len(parts[i].split())
    meta = make_metadata(topic, outline, chapters if len(chapters) >= 3 else [])
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    for k, pos in enumerate((0.15, 0.45, 0.75)):
        if thumb_done[k]:
            continue
        make_thumbnail(imgs[min(int(len(imgs) * pos), len(imgs) - 1)], topic, d / f"thumbnail_{'abc'[k]}.jpg",
                       variant=k, minutes=int(total // 60))

    for f in list(d.glob("voice_*")) + list(d.glob("sec_*.txt")) + [d / "voice_list.txt", d / "outline.json",
                                                                      d / "prompts.json", d / "t.srt", d / "wordtimes.json"]:
        f.unlink(missing_ok=True)
    with open(ROOT / "done.txt", "a") as f:
        f.write(topic + "\n")
    log("DONE ->", d / "video.mp4")


if __name__ == "__main__":
    main()
