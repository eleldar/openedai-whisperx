import gc
import logging
import os
from typing import List, Optional
from uuid import uuid4

import torch
import whisperx
from fastapi import Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import openedai
from config import state

logger = logging.getLogger(__name__)

app = openedai.OpenAIStub()


def save_file(file: UploadFile) -> str:
    tempfile = state.tempfiles / str(uuid4())
    with open(tempfile, "wb") as f:
        f.write(file.file.read())
    return str(tempfile)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    model: str = Form(state.model),
    language: Optional[str] = Form(state.language),
    prompt: Optional[str] = Form(state.prompt),
    temperature: Optional[float] = Form(state.temperature),
    response_format: Optional[str] = Form(state.response_format),
    timestamp_granularities: Optional[List[str]] = Form([state.timestamp_granularities]),
):
    try:
        tempfile = save_file(file)
    except Exception as error:
        tempfile = None
        logger.error(error)
        raise HTTPException(status_code=404, detail=error)

    try:
        model = whisperx.load_model(
            state.model_mapping.get(model),
            device=state.device,
            compute_type=state.compute_type,
            asr_options={"temperatures": temperature, "initial_prompt": prompt},
            download_root=state.model_dir,
        )
        audio = whisperx.load_audio(tempfile)
        result = model.transcribe(audio, batch_size=state.batch_size, language=language, task="transcribe")
        gc.collect()
        torch.cuda.empty_cache()
        del model
        model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=state.device, model_dir=state.model_dir
        )
        result = whisperx.align(
            result["segments"], model, metadata, audio, state.device, return_char_alignments=False, print_progress=True
        )
        gc.collect()
        torch.cuda.empty_cache()
        del model
    except Exception as error:
        logger.error(error)
        raise HTTPException(status_code=404, detail=error)
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
            "duration": result["segments"][-1].get("end") if result.get("segments") else [],
            "language": language if language else state.language,
            "text": (
                " ".join(
                    [
                        segment.get("text", "").strip()
                        for segment in result["segments"]
                        if segment.get("text", "").strip()
                    ]
                )
                if result.get("segments")
                else ""
            ),
            "segments": result["segments"] if result.get("segments") else [],
            "words": result["word_segments"] if result.get("word_segments") else [],
            "word_segments": result["word_segments"] if result.get("word_segments") else [],
        }
        return JSONResponse(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={file.filename}_verbose.json"},
        )
    except Exception as error:
        logger.error(error)
        raise HTTPException(status_code=404, detail=error)
