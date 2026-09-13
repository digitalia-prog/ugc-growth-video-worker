import os
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
Ne laisse aucun champ vide.

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

    return {
        "transcript": parsed.get("transcript") or transcript,
        "summary": parsed.get("summary") or "Résumé non détecté.",
        "hook": parsed.get("hook") or "Hook non détecté.",
        "structure": parsed.get("structure") or "Structure non détectée.",
        "angle": parsed.get("angle") or "Angle marketing non détecté.",
        "psychology": parsed.get("psychology") if isinstance(parsed.get("psychology"), list) else ["Curiosité", "Identification"],
        "strengths": parsed.get("strengths") if isinstance(parsed.get("strengths"), list) else ["Contenu engageant", "Sujet clair"],
        "weaknesses": parsed.get("weaknesses") if isinstance(parsed.get("weaknesses"), list) else ["Hook à renforcer", "CTA à clarifier"],
        "recreateIdeas": parsed.get("recreateIdeas") if isinstance(parsed.get("recreateIdeas"), list) else ["Recréer la vidéo avec une structure plus claire"],
        "similarHooks": parsed.get("similarHooks") if isinstance(parsed.get("similarHooks"), list) else ["Tu ne vas pas croire ce qui se passe ici"],
        "similarAngles": parsed.get("similarAngles") if isinstance(parsed.get("similarAngles"), list) else ["Angle curiosité", "Angle preuve sociale"],
        "scriptPrompt": parsed.get("scriptPrompt") or "Créer une vidéo UGC courte avec hook fort, preuve, démonstration et CTA.",
        "viralScore": parsed.get("viralScore") or "6/10 — potentiel correct à optimiser.",
        "whyItWorks": parsed.get("whyItWorks") if isinstance(parsed.get("whyItWorks"), list) else ["Le format peut créer de la curiosité", "Le sujet est facile à comprendre"],
        "howToBeat": parsed.get("howToBeat") if isinstance(parsed.get("howToBeat"), list) else ["Ajouter un hook plus direct", "Renforcer le CTA final"],
        "adsAngles": parsed.get("adsAngles") if isinstance(parsed.get("adsAngles"), list) else ["Angle problème/solution", "Angle curiosité", "Angle preuve"],
        "creativeType": parsed.get("creativeType") or "UGC / contenu social",
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
