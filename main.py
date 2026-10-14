import os
import tempfile
import subprocess
import base64
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("OPENAI_API_KEY manquante sur Railway")

client = OpenAI(api_key=api_key)

UPLOAD_CHUNK_BYTES = 1024 * 1024
# Garde une marge sous la limite Whisper de 25 MiB.
MAX_TRANSCRIPTION_BYTES = 24 * 1024 * 1024


async def save_upload(upload: UploadFile, destination: Path):
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)


def extract_audio_for_transcription(input_path: Path, tmpdir: str) -> Path:
    output_path = Path(tmpdir) / "transcription-audio.mp3"

    def encode(bitrate: str):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                bitrate,
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )

    try:
        encode("48k")
        if output_path.stat().st_size > MAX_TRANSCRIPTION_BYTES:
            encode("24k")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Audio extraction timeout")
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "unknown ffmpeg error")[-1200:]
        raise HTTPException(
            status_code=422,
            detail=f"Audio extraction failed: {details}",
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(status_code=422, detail="Aucune piste audio exploitable")

    if output_path.stat().st_size > MAX_TRANSCRIPTION_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio trop long apres compression pour une transcription unique",
        )

    return output_path


def transcribe_uploaded_media(input_path: Path, tmpdir: str) -> str:
    audio_path = extract_audio_for_transcription(input_path, tmpdir)

    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    return transcription.text or ""


class VideoRequest(BaseModel):
    url: str


class VisualAnalysisRequest(BaseModel):
    url: str
    transcript: str = ""
    language: str = "French"
    context: dict = Field(default_factory=dict)


MAX_VISUAL_SECONDS = 180.0
MAX_VISUAL_FRAMES = 10


def _format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - (minutes * 60)
    return f"{minutes:02d}:{secs:05.2f}"


def _download_video_for_visuals(url: str, tmpdir: str) -> Path:
    output_template = str(Path(tmpdir) / "visual-source.%(ext)s")
    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "best[height<=720]/best",
        "-o",
        output_template,
        url,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Téléchargement vidéo trop long")
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "yt-dlp error")[-1200:]
        raise HTTPException(
            status_code=422,
            detail=f"Vidéo inaccessible pour l’analyse visuelle: {details}",
        )

    candidates = sorted(Path(tmpdir).glob("visual-source.*"))
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail="Aucun fichier vidéo exploitable pour l’analyse visuelle",
        )
    return candidates[0]


def _probe_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        return max(0.0, float((result.stdout or "0").strip() or 0))
    except Exception:
        return 0.0


def _detect_scene_timestamps(video_path: Path, duration: float) -> list[float]:
    scan_limit = min(duration or MAX_VISUAL_SECONDS, MAX_VISUAL_SECONDS)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-t",
        f"{scan_limit:.3f}",
        "-i",
        str(video_path),
        "-vf",
        "select='gt(scene,0.30)',showinfo",
        "-vsync",
        "vfr",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        times = [
            float(value)
            for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr or "")
        ]
    except Exception:
        times = []

    end = scan_limit if scan_limit > 0 else 30.0
    candidates = [0.2]
    candidates.extend(t for t in times if 0.2 <= t <= end)

    # Si le montage contient peu de coupes détectables, compléter par des échantillons
    # temporels réels afin de couvrir le début, le milieu et la fin.
    if len(candidates) < 5 and end > 1:
        for fraction in (0.15, 0.35, 0.55, 0.75, 0.92):
            candidates.append(max(0.2, min(end - 0.1, end * fraction)))

    # Dédupliquer les captures trop proches.
    deduped = []
    for value in sorted(candidates):
        if not deduped or abs(value - deduped[-1]) >= 0.65:
            deduped.append(value)

    if len(deduped) > MAX_VISUAL_FRAMES:
        indexes = [
            round(i * (len(deduped) - 1) / (MAX_VISUAL_FRAMES - 1))
            for i in range(MAX_VISUAL_FRAMES)
        ]
        deduped = [deduped[i] for i in indexes]

    return deduped[:MAX_VISUAL_FRAMES]


def _extract_frame_data_url(video_path: Path, timestamp: float, output_path: Path) -> str:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=720:-2:force_original_aspect_ratio=decrease",
                "-q:v",
                "5",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )
    except Exception:
        return ""

    if not output_path.exists() or output_path.stat().st_size == 0:
        return ""
    encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_visual_result(parsed: dict, frames: list[dict], duration: float) -> dict:
    valid_frames = {index + 1: frame for index, frame in enumerate(frames)}

    timeline = []
    for raw in parsed.get("visualTimeline", []) if isinstance(parsed, dict) else []:
        if not isinstance(raw, dict):
            continue
        try:
            frame_index = int(raw.get("frameIndex"))
        except Exception:
            continue
        frame = valid_frames.get(frame_index)
        if not frame:
            continue
        assessment = raw.get("assessment")
        if assessment not in {"strong", "medium", "weak", "neutral"}:
            assessment = "neutral"
        risk = raw.get("attentionRisk")
        if risk not in {"low", "medium", "high", "unknown"}:
            risk = "unknown"
        timeline.append(
            {
                "frameIndex": frame_index,
                "timestampSeconds": frame["timestampSeconds"],
                "timecode": frame["timecode"],
                "label": str(raw.get("label") or "Moment analysé"),
                "observation": str(raw.get("observation") or "Observation non déterminable"),
                "assessment": assessment,
                # Le risque d’attention est une estimation créative, jamais une courbe de rétention réelle.
                "attentionRisk": risk,
                "evidence": "observed",
            }
        )

    cta = parsed.get("ctaAnalysis") if isinstance(parsed, dict) else {}
    if not isinstance(cta, dict):
        cta = {}
    cta_frame = None
    try:
        cta_frame = int(cta.get("frameIndex")) if cta.get("frameIndex") is not None else None
    except Exception:
        cta_frame = None
    cta_anchor = valid_frames.get(cta_frame) if cta_frame else None
    cta_source = cta.get("evidenceSource")
    if cta_source not in {"visual", "audio", "imported", "none"}:
        cta_source = "none"
    cta_status = cta.get("status")
    if cta_status not in {"explicit", "implicit", "absent", "unknown"}:
        cta_status = "unknown"

    priority_fixes = []
    for index, raw in enumerate(parsed.get("priorityFixes", []) if isinstance(parsed, dict) else [], start=1):
        if not isinstance(raw, dict) or len(priority_fixes) >= 3:
            continue
        priority_fixes.append(
            {
                "priority": len(priority_fixes) + 1,
                "diagnosis": str(raw.get("diagnosis") or ""),
                "evidence": str(raw.get("evidence") or ""),
                "action": str(raw.get("action") or ""),
                "test": str(raw.get("test") or ""),
            }
        )

    def safe_object(name):
        value = parsed.get(name) if isinstance(parsed, dict) else {}
        return value if isinstance(value, dict) else {}

    return {
        "visualAnalysisAvailable": True,
        "visualAnalysisNote": "Analyse basée sur des captures réelles horodatées de la vidéo. Les risques d’attention sont des estimations créatives, pas des données de rétention plateforme.",
        "durationSeconds": round(duration, 2),
        "frameCount": len(frames),
        "visualTimeline": timeline,
        "visualHook": safe_object("visualHook"),
        "pacing": safe_object("pacing"),
        "proofAnalysis": safe_object("proofAnalysis"),
        "ctaAnalysis": {
            "status": cta_status,
            "text": str(cta.get("text") or ""),
            "frameIndex": cta_frame if cta_anchor else None,
            "timestampSeconds": cta_anchor["timestampSeconds"] if cta_anchor else None,
            "timecode": cta_anchor["timecode"] if cta_anchor else "",
            "evidenceSource": cta_source,
            "explanation": str(cta.get("explanation") or ""),
        },
        "onScreenText": [
            str(item).strip()
            for item in parsed.get("onScreenText", [])
            if isinstance(item, str) and item.strip()
        ][:12] if isinstance(parsed, dict) and isinstance(parsed.get("onScreenText"), list) else [],
        "priorityFixes": priority_fixes,
        "dataIntegrity": {
            "observed": [
                "Images réelles extraites de la vidéo avec horodatages",
            ],
            "estimated": [
                "Risque de décrochage / attention à partir du rythme, du contenu et du montage"
            ],
            "imported": ["Métadonnées publicitaires fournies par la source"],
            "unavailableMetrics": [
                "CTR réel",
                "ROAS réel",
                "Taux de conversion réel",
                "Courbe de rétention plateforme réelle",
            ],
        },
    }


def _analyze_visual_frames(frames: list[dict], transcript: str, language: str, context: dict, duration: float) -> dict:
    frame_legend = "\n".join(
        f"FRAME {index + 1}: {frame['timecode']} ({frame['timestampSeconds']:.2f}s)"
        for index, frame in enumerate(frames)
    )
    prompt = f"""
You are a senior creative strategist specialized in UGC and performance creative.
Return the full JSON analysis in {language}.

You are given REAL timestamped frames extracted from one video, plus its real transcript and imported metadata.

STRICT EVIDENCE POLICY:
- Never invent a CTA, product, person, on-screen text, proof, shot change, metric, performance result or platform data.
- A visual claim must be supported by one of the supplied frames.
- An audio/verbal claim must be supported by the transcript.
- Imported metadata is not visual evidence. If a CTA exists only in metadata, label evidenceSource as "imported".
- If the supplied frames do not prove that a CTA is absent from the entire video, use status "unknown", not "absent".
- You do NOT have CTR, ROAS, conversion rate or a real platform retention curve. Never invent them.
- attentionRisk is only a CREATIVE ESTIMATE based on visible pacing/information density and must never be described as measured retention.
- Do not invent confidence percentages.
- For visualTimeline, use ONLY frameIndex values listed below. Do not create timecodes yourself.
- Give at most 3 priority fixes. Every fix must cite concrete evidence from a frame, transcript, or imported metadata.

FRAME LEGEND:
{frame_legend}

REAL TRANSCRIPT:
<<<TRANSCRIPT>>>
{transcript}
<<<END TRANSCRIPT>>>

IMPORTED METADATA:
{json.dumps(context, ensure_ascii=False)}

Return valid JSON only with this exact shape:
{{
  "visualTimeline": [
    {{
      "frameIndex": 1,
      "label": "",
      "observation": "",
      "assessment": "strong|medium|weak|neutral",
      "attentionRisk": "low|medium|high|unknown"
    }}
  ],
  "visualHook": {{
    "observation": "",
    "assessment": "strong|medium|weak|neutral",
    "evidence": "what in the supplied frames/transcript supports this"
  }},
  "pacing": {{
    "observation": "",
    "assessment": "strong|medium|weak|neutral",
    "evidence": ""
  }},
  "proofAnalysis": {{
    "status": "observed|not_observed|unknown",
    "observation": "",
    "evidenceSource": "visual|audio|imported|none"
  }},
  "ctaAnalysis": {{
    "status": "explicit|implicit|absent|unknown",
    "text": "",
    "frameIndex": null,
    "evidenceSource": "visual|audio|imported|none",
    "explanation": ""
  }},
  "onScreenText": [],
  "priorityFixes": [
    {{
      "diagnosis": "",
      "evidence": "",
      "action": "",
      "test": ""
    }}
  ]
}}
"""

    content = [{"type": "text", "text": prompt}]
    for index, frame in enumerate(frames):
        content.append(
            {
                "type": "text",
                "text": f"FRAME {index + 1} — {frame['timecode']}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame["dataUrl"], "detail": "high"},
            }
        )

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_VISION_MODEL") or os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4.1-mini"),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "Evidence-first creative analysis. Never fabricate observations or performance metrics. Return valid JSON only.",
            },
            {"role": "user", "content": content},
        ],
    )

    raw = completion.choices[0].message.content or "{}"
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}
    return _normalize_visual_result(parsed, frames, duration)


@app.post("/visual-analyze")
def visual_analyze(payload: VisualAnalysisRequest):
    if not payload.url:
        raise HTTPException(status_code=400, detail="Missing url")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = _download_video_for_visuals(payload.url, tmpdir)
        duration = _probe_duration(video_path)
        timestamps = _detect_scene_timestamps(video_path, duration)

        frames = []
        frame_dir = Path(tmpdir) / "visual-frames"
        frame_dir.mkdir(exist_ok=True)
        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = frame_dir / f"frame-{index:02d}.jpg"
            data_url = _extract_frame_data_url(video_path, timestamp, frame_path)
            if not data_url:
                continue
            frames.append(
                {
                    "timestampSeconds": round(timestamp, 3),
                    "timecode": _format_timecode(timestamp),
                    "dataUrl": data_url,
                }
            )

        if not frames:
            raise HTTPException(
                status_code=422,
                detail="Aucune image exploitable n’a pu être extraite de la vidéo",
            )

        return _analyze_visual_frames(
            frames=frames,
            transcript=payload.transcript or "",
            language=payload.language or "French",
            context=payload.context or {},
            duration=duration,
        )


def clean_text(value):
    return value.strip() if isinstance(value, str) and value.strip() else "-"


def analyze_transcript(transcript, platform="-", product="-", audience="-", notes="-"):
    prompt = f"""
Tu es un expert senior en UGC ads, creative strategy et performance marketing.

Analyse cette vidéo comme une agence marketing professionnelle.

Contexte :
- Plateforme : {platform}
- Produit / Offre : {product}
- Audience : {audience}
- Notes : {notes}

Transcript réel :
\"\"\"
{transcript}
\"\"\"

Réponds uniquement en JSON valide, sans markdown.
N’invente aucun CTA, score, preuve, métrique ou élément visuel qui n’est pas dans le transcript ou le contexte.
Le champ viralScore doit rester vide, car aucune donnée de performance réelle n’est fournie.

Format :
{{
  "transcript": "",
  "summary": "",
  "hook": "",
  "structure": "",
  "angle": "",
  "psychology": ["", ""],
  "strengths": ["", ""],
  "weaknesses": ["", ""],
  "recreateIdeas": ["", ""],
  "similarHooks": ["", ""],
  "similarAngles": ["", ""],
  "scriptPrompt": "",
  "viralScore": "",
  "whyItWorks": ["", ""],
  "howToBeat": ["", ""],
  "adsAngles": ["", ""],
  "creativeType": ""
}}
"""

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4.1-mini"),
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": "Tu es un expert marketing UGC. Tu réponds uniquement en JSON valide complet.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    import json
    raw = completion.choices[0].message.content or "{}"
    raw = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    def string_list(name):
        value = parsed.get(name)
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    return {
        "transcript": transcript,
        "summary": parsed.get("summary") or "Analyse non disponible avec les éléments reçus.",
        "hook": parsed.get("hook") or "Hook non déterminable avec les éléments reçus.",
        "structure": parsed.get("structure") or "Structure non déterminable avec les éléments reçus.",
        "angle": parsed.get("angle") or "Angle non déterminable avec les éléments reçus.",
        "psychology": string_list("psychology"),
        "strengths": string_list("strengths"),
        "weaknesses": string_list("weaknesses"),
        "recreateIdeas": string_list("recreateIdeas"),
        "similarHooks": string_list("similarHooks"),
        "similarAngles": string_list("similarAngles"),
        "scriptPrompt": parsed.get("scriptPrompt") or "",
        "viralScore": "",
        "whyItWorks": string_list("whyItWorks"),
        "howToBeat": string_list("howToBeat"),
        "adsAngles": string_list("adsAngles"),
        "creativeType": parsed.get("creativeType") or "Créatif non déterminé",
    }


@app.get("/")
def health():
    return {"ok": True, "service": "ugc-growth-video-worker"}


@app.post("/transcribe")
def transcribe_video(payload: VideoRequest):
    if not payload.url:
        raise HTTPException(status_code=400, detail="Missing url")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = str(Path(tmpdir) / "audio.%(ext)s")
        audio_path = str(Path(tmpdir) / "audio.mp3")

        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-x",
                    "--audio-format",
                    "mp3",
                    "--audio-quality",
                    "0",
                    "-o",
                    output_template,
                    payload.url,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )

            if not Path(audio_path).exists():
                files = list(Path(tmpdir).glob("audio.*"))
                if not files:
                    raise HTTPException(status_code=500, detail="Audio extraction failed")
                audio_path = str(files[0])

            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )

            return {"transcript": transcript.text, "sourceUrl": payload.url}

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Video download timeout")
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"yt-dlp failed: {e.stderr[-1000:] if e.stderr else 'unknown error'}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-transcribe")
async def upload_transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Missing file")

    with tempfile.TemporaryDirectory() as tmpdir:
        safe_name = Path(file.filename or "upload.bin").name
        input_path = Path(tmpdir) / safe_name

        try:
            await save_upload(file, input_path)
            transcript = transcribe_uploaded_media(input_path, tmpdir)

            if not transcript.strip():
                raise HTTPException(status_code=422, detail="Transcript vide")

            return {
                "success": True,
                "noStorage": True,
                "filename": safe_name,
                "transcript": transcript,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload-analyze")
async def upload_analyze(
    file: UploadFile = File(...),
    platform: str = Form("-"),
    product: str = Form("-"),
    audience: str = Form("-"),
    notes: str = Form("-"),
):
    if not file:
        raise HTTPException(status_code=400, detail="Missing file")

    with tempfile.TemporaryDirectory() as tmpdir:
        safe_name = Path(file.filename or "upload.bin").name
        input_path = Path(tmpdir) / safe_name

        try:
            await save_upload(file, input_path)
            transcript = transcribe_uploaded_media(input_path, tmpdir)

            if not transcript.strip():
                raise HTTPException(status_code=422, detail="Transcript vide")

            result = analyze_transcript(
                transcript=transcript,
                platform=clean_text(platform),
                product=clean_text(product),
                audience=clean_text(audience),
                notes=clean_text(notes),
            )

            return {
                "success": True,
                "noStorage": True,
                "filename": safe_name,
                **result,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
