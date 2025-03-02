#!/usr/bin/env python3
import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import torch
import uvicorn
import whisperx
from fastapi import Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import openedai

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

app = openedai.OpenAIStub()

model_mapping = {
    "openai/whisper-tiny": "tiny",
    "openai/whisper-base": "base",
    "openai/whisper-small": "small",
    "openai/whisper-medium": "medium",
    "openai/whisper-large": "large-v1",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}
state = {
    "model": "openai/whisper-large-v3",
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "compute_type": "float16" if torch.cuda.is_available() else "int8",
    "temperature": 0,
    "language": "ru",
    "prompt": None,
    "response_format": "verbose_json",
    "timestamp_granularities": "word",
}
tempfiles = Path("/app/tempfiles")
model_dir = Path("/app/models")


def srt_time(t):
    return "{:02d}:{:02d}:{:06.3f}".format(int(t // 3600), int(t // 60) % 60, t % 60).replace(".", ",")


def save_file(file: UploadFile) -> str:
    tempfile = tempfiles / str(uuid4())
    with open(tempfile, "wb") as f:
        f.write(file.file.read())
    return str(tempfile)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    model: str = Form(state["model"]),
    language: Optional[str] = Form(state["language"]),
    prompt: Optional[str] = Form(state["prompt"]),
    temperature: Optional[float] = Form(state["temperature"]),
    response_format: Optional[str] = Form(state["response_format"]),
    timestamp_granularities: Optional[List[str]] = Form([state["timestamp_granularities"]]),
):
    try:
        tempfile = save_file(file)
    except Exception as error:
        tempfile = None
        logger.error(error)
        raise HTTPException(status_code=404, detail="Error save file")

    try:
        model = whisperx.load_model(
            model_mapping.get(model),
            device=state["device"],
            compute_type=state["compute_type"],
            asr_options={"temperatures": temperature, "initial_prompt": prompt},
            download_root=model_dir,
        )
        audio = whisperx.load_audio(tempfile)
        result = model.transcribe(audio, batch_size=state["batch_size"], language=language, task="transcribe")
        gc.collect()
        torch.cuda.empty_cache()
        del model
        model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=state["device"], model_dir=model_dir
        )
        result = whisperx.align(
            result["segments"], model, metadata, audio, device, return_char_alignments=False, print_progress=True
        )
        gc.collect()
        torch.cuda.empty_cache()
        del model
    except Exception as error:
        logger.error(error)
        raise HTTPException(status_code=404, detail="Error transcribation")
    finally:
        if tempfile and os.path.exists(tempfile):
            os.remove(tempfile)
        gc.collect()
        torch.cuda.empty_cache()
        if "model" in locals() or "model" in globals():
            del model
        gc.collect()

    try:
        content = {
            "duration": result["segments"][-1].get("end"),
            "language": language,
            "text": " ".join(
                [segment.get("text", "").strip() for segment in result["segments"] if segment.get("text", "").strip()]
            ),
            "segments": result["segments"],
            "words": result["word_segments"],
            "word_segments": result["word_segments"],
            "align_result": result,
        }
        return JSONResponse(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={file.filename}_verbose.json"},
        )
    except Exception as error:
        logger.error(error)
        raise HTTPException(status_code=404, detail="Error create JSON")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="whisper.py",
        description="OpenedAI Whisper API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        action="store",
        default="openai/whisper-large-v3",
        help="The model to use for transcription. Ex. distil-whisper/medium",
    )
    parser.add_argument(
        "-d", "--device", action="store", default="auto", help="Set the torch device for the model. Ex. cuda:1"
    )
    parser.add_argument(
        "-t",
        "--dtype",
        action="store",
        default="auto",
        help="Set the torch data type for processing (float16, int8)",
    )
    parser.add_argument("-P", "--port", action="store", default=8000, type=int, help="Server tcp port")
    parser.add_argument("-H", "--host", action="store", default="localhost", help="Host to listen on, Ex. 0.0.0.0")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    for _name, _model in model_mapping.items():
        app.register_model(name=_name, model=_model)
    args = parse_args(sys.argv[1:])
    state["model"] = args.model
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        compute_type = "float16" if torch.cuda.is_available() else "int8"
    uvicorn.run(
        app, host=args.host, port=args.port
    )  # , root_path=cwd, access_log=False, log_level="info", ssl_keyfile="cert.pem", ssl_certfile="cert.pem")
