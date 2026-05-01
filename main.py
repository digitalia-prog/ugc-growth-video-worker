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
        safe_name = file.filename or "upload.mp4"
        input_path = Path(tmpdir) / safe_name

        try:
            content = await file.read()
            input_path.write_bytes(content)

            with open(input_path, "rb") as uploaded_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=uploaded_file,
                )

            transcript = transcription.text or ""

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
