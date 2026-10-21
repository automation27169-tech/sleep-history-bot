#!/usr/bin/env python3
"""Topic -> script -> voice -> images -> video. Free stack: Gemini + Pollinations + FFmpeg."""
import os, sys, re, json, textwrap, time, wave, math, random, shutil, asyncio, subprocess, urllib.parse
from pathlib import Path
import requests
from google import genai
from google.genai import types

API_KEY = os.environ["GEMINI_API_KEY"]
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-flash-latest")
TTS_MODEL = os.getenv("TTS_MODEL", "")   # optional override; otherwise TTS_MODELS below are tried in order
VOICE = os.getenv("VOICE", "Algieba")              # one fixed voice for the whole channel
EDGE_VOICE = os.getenv("EDGE_VOICE", "en-US-GuyNeural")
SECTIONS = int(os.getenv("SECTIONS", "8"))          # 8 sections ~ 45-50 min of audio
WORDS_PER_SECTION = int(os.getenv("WORDS_PER_SECTION", "800"))
IMAGE_SECONDS = int(os.getenv("IMAGE_SECONDS", "40"))
FPS = 15
STYLE = os.getenv("STYLE", "soft painterly digital illustration, warm candlelight, deep blue and amber tones, "
                           "serene, dim lighting, wide landscape composition, no text, no close-up faces")
NARRATION_STYLE = ("Read this as a sleep story narrator: very slow, soft, warm, calm and soothing, "
                   "with long gentle pauses between sentences: ")

client = genai.Client(api_key=API_KEY)
ROOT = Path(__file__).parent


def discover_models():
    """Ask Google which models THIS api key can use, so we never call a model that doesn't exist."""
    text, tts = [], []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if acts and "generateContent" not in acts:
                continue
            low = name.lower()
            if "tts" in low:
                tts.append(name)
            elif "flash" in low and not any(x in low for x in
                    ("image", "live", "audio", "native", "embedding", "robotics", "computer", "exp", "vision")):
                text.append(name)
    except Exception as e:
        print("Could not list models:", str(e)[:150], flush=True)
    order = lambda n: (("preview" in n), ("lite" in n), n)
    text = sorted(set(text), key=order)[:4]
    tts = sorted(set(tts), key=order)[:4]
    return text, tts


_text, _tts = discover_models()
TEXT_MODELS = list(dict.fromkeys([TEXT_MODEL] + _text)) or ["gemini-2.5-flash"]
TTS_MODELS = list(dict.fromkeys([m for m in [TTS_MODEL] if m] + _tts)) or ["gemini-2.5-flash-preview-tts"]
print("TEXT models to try:", TEXT_MODELS, flush=True)
print("TTS models to try:", TTS_MODELS, flush=True)


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
        outline = json.loads(ask(
            f'Plan a calm, factual history sleep story titled "{topic}". Return a JSON list of exactly {SECTIONS} '
            f'objects with keys "title" and "summary" (1 sentence). The story drifts slowly through one place and '
            f'time period, moving from gentle arrival to deep rest.', json_out=True))
        of.write_text(json.dumps(outline))
    parts, prev_tail = [], ""
    for i, sec in enumerate(outline, 1):
        sf = d / f"sec_{i:02d}.txt"
        if sf.exists():
            text = sf.read_text()
        else:
            log(f"Section {i}/{len(outline)}: {sec['title']}")
            last = (i == len(outline))
            text = ask(f"""You are writing section {i} of {len(outline)} of a calm history sleep story: "{topic}".
Full outline: {json.dumps(outline)}
This section: {sec['title']} - {sec['summary']}
Previous section ended with: "{prev_tail}"

Rules:
- About {WORDS_PER_SECTION} words, plain paragraphs only, no headings, no markdown, no lists.
- Second person, present tense ("You walk..."). Short, slow, gentle sentences. Rich sensory detail (light, sound, smell, warmth).
- Historically grounded. Only state details you are confident are accurate; when unsure, stay general. No invented statistics or dates dumps.
- Nothing violent, frightening, or jarring. No cliffhangers.
- Continue naturally from the previous section without recapping it.
{"- This is the opening section: begin with a short, warm welcome to the listener (settle in, dim the lights, breathe slowly), then gently begin the journey." if i == 1 else ""}
{"- This is the final section: slow down further and end with a soft, quiet fade into sleep." if last else ""}""")
            sf.write_text(text)
        parts.append(text)
        prev_tail = text[-250:]
    return outline, "\n\n".join(parts)


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
    run(["ffmpeg", "-y", "-i", str(raw), "-af", "loudnorm=I=-20:LRA=7:TP=-2,apad=pad_dur=6", str(out)])
    raw.unlink(missing_ok=True)
    return out


# ---------- 3. IMAGES ----------
def scene_prompts(topic, outline, n, d):
    pf = d / "prompts.json"
    if pf.exists():
        cached = json.loads(pf.read_text())
        if len(cached) == n:
            return cached
    prompts, batch = [], 30
    for start in range(0, n, batch):
        end = min(n, start + batch)
        txt = ask(f"""Story: "{topic}". Outline: {json.dumps(outline)}
Write image prompts number {start+1} to {end} of {n} for illustrations that follow the story in order from beginning to end
(so these cover roughly {int(start/n*100)}% to {int(end/n*100)}% of the story).
Each prompt: one vivid scene, max 25 words, calm and peaceful, no text, no modern objects, no close-up faces.
Return a JSON list of exactly {end-start} strings.""", json_out=True)
        got = json.loads(txt)
        got = (got + got[-1:] * (end - start))[: end - start] if got else ["a quiet moonlit ancient landscape"] * (end - start)
        prompts += got
    pf.write_text(json.dumps(prompts))
    return prompts


def placeholder(path):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x0b1530:s=1280x720", "-frames:v", "1", str(path)])


def make_image(prompt, path, seed):
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(f"{prompt}, {STYLE}") +
           f"?width=1280&height=720&seed={seed}&nologo=true&model=flux")
    for t in range(4):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 5000:
                path.write_bytes(r.content)
                return True
        except Exception as e:
            log("  image error:", str(e)[:100])
        time.sleep(15 * (t + 1))
    return False


def make_images(topic, outline, n, d):
    prompts = scene_prompts(topic, outline, n, d)
    imgs, last_ok = [], None
    for i, p in enumerate(prompts):
        path = d / f"img_{i:03d}.jpg"
        if path.exists() and path.stat().st_size > 1000:
            imgs.append(path); last_ok = path; continue
        log(f"Image {i+1}/{n}")
        if make_image(p, path, seed=random.randint(1, 10**6)):
            last_ok = path
        elif last_ok:
            shutil.copy(last_ok, path)
        else:
            placeholder(path)
        imgs.append(path)
        time.sleep(6)   # be gentle with the free service
    return imgs


# ---------- 4. VIDEO ----------
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)]).strip())


def render_clip(img, out, secs):
    frames = int(secs * FPS)
    if random.random() < 0.5:
        z = "min(zoom+0.0003,1.2)"
    else:
        z = "if(eq(on,0),1.2,max(zoom-0.0003,1.02))"
    vf = (f"scale=2560:1440,zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps={FPS},"
          f"fade=t=in:st=0:d=1.5,fade=t=out:st={max(secs-1.5,0):.2f}:d=1.5,format=yuv420p")
    run(["ffmpeg", "-y", "-i", str(img), "-vf", vf, "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "faster", "-crf", "24", str(out)])


def make_video(imgs, narration, out, d):
    total = duration(narration)
    secs = total / len(imgs)
    clips = []
    for i, img in enumerate(imgs):
        c = d / f"clip_{i:03d}.mp4"
        if not c.exists():
            log(f"Clip {i+1}/{len(imgs)}")
            render_clip(img, c, secs)
        clips.append(c)
    lst = d / "clips.txt"
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    music = next(iter((ROOT / "assets").glob("music.*")), None)
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(narration)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music), "-filter_complex",
                "[2:a]volume=0.08[m];[1:a][m]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    run(cmd)
    for c in clips:
        c.unlink()


# ---------- 5. METADATA ----------
def make_metadata(topic, outline):
    m = json.loads(ask(f"""YouTube metadata for a calm history sleep story about "{topic}" (US audience).
Return JSON with keys: "title" (max 70 chars, like "Fall Asleep to {topic} | Calm History Sleep Story"),
"description" (3-4 calm sentences + short chapter list based on this outline: {json.dumps(outline)}),
"tags" (list of 12 short tags).""", json_out=True))
    m["description"] += "\n\nNarration and visuals in this video are AI-generated. Educational, relaxing history storytelling."
    return m


# ---------- 6. THUMBNAIL ----------
def make_thumbnail(img, topic, out, d):
    try:
        tf = d / "thumb_text.txt"
        tf.write_text("\n".join(textwrap.wrap(topic.upper(), 18)))
        vf = ("scale=1280:720,eq=brightness=-0.10:saturation=1.1,"
              "drawbox=x=0:y=ih*0.52:w=iw:h=ih*0.48:color=black@0.5:t=fill,"
              f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile={tf}:"
              "fontcolor=white:fontsize=80:x=(w-text_w)/2:y=h*0.60:line_spacing=14:borderw=3:bordercolor=black")
        run(["ffmpeg", "-y", "-i", str(img), "-vf", vf, "-frames:v", "1", "-q:v", "2", str(out)])
        tf.unlink(missing_ok=True)
    except Exception as e:
        log("Thumbnail text failed, using plain image:", str(e)[:150])
        shutil.copy(img, out)


# ---------- MAIN ----------
def pick_topic():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    done = set((ROOT / "done.txt").read_text().splitlines()) if (ROOT / "done.txt").exists() else set()
    for line in (ROOT / "topics.txt").read_text().splitlines():
        if line.strip() and line.strip() not in done:
            return line.strip()
    sys.exit("No topics left in topics.txt")


def main():
    topic = pick_topic()
    log("TOPIC:", topic)
    d = ROOT / "output" / slugify(topic)
    d.mkdir(parents=True, exist_ok=True)

    outline, script = write_script(topic, d)
    (d / "script.txt").write_text(script)
    narration = make_voice(script, d)
    n = math.ceil(duration(narration) / IMAGE_SECONDS)
    imgs = make_images(topic, outline, n, d)
    make_video(imgs, narration, d / "video.mp4", d)
    meta = make_metadata(topic, outline)
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    make_thumbnail(imgs[min(2, len(imgs) - 1)], topic, d / "thumbnail.jpg", d)
    for f in list(d.glob("voice_*")) + [d / "voice_list.txt", d / "clips.txt"]:
        f.unlink(missing_ok=True)
    for f in list(d.glob("sec_*.txt")) + [d / "outline.json", d / "prompts.json"]:
        f.unlink(missing_ok=True)
    with open(ROOT / "done.txt", "a") as f:
        f.write(topic + "\n")
    log("DONE ->", d / "video.mp4")


if __name__ == "__main__":
    main()
