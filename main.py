import os
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class VideoRequest(BaseModel):
    url: str

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

            return {
                "transcript": transcript.text,
                "sourceUrl": payload.url,
            }

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Video download timeout")
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"yt-dlp failed: {e.stderr[-1000:] if e.stderr else 'unknown error'}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
