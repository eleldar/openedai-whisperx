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
        raise HTTPException(status_code=404, detail="Error save file")

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
            "language": language if language else state.language,
            "text": " ".join(
                [segment.get("text", "").strip() for segment in result["segments"] if segment.get("text", "").strip()]
            ),
            "segments": result["segments"],
            "words": result["word_segments"],
            "word_segments": result["word_segments"],
        }
        return JSONResponse(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={file.filename}_verbose.json"},
        )
    except Exception as error:
        logger.error(error)
        raise HTTPException(status_code=404, detail="Error create JSON")


# TO-DO
# async def whisper(file, response_format: str, **kwargs):
#     global pipe
#
#     result = pipe(await file.read(), **kwargs)
#
#     filename_noext, ext = os.path.splitext(file.filename)
#
#     if response_format == "text":
#         return PlainTextResponse(
#             result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"}
#         )
#
#     elif response_format == "json":
#         return JSONResponse(
#             content={"text": result["text"].strip()},
#             media_type="application/json",
#             headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"},
#         )
#
#     elif response_format == "verbose_json":
#         chunks = result["chunks"]
#
#         response = {
#             "task": kwargs["generate_kwargs"]["task"],
#             # "language": "english",
#             "duration": chunks[-1]["timestamp"][1],
#             "text": result["text"].strip(),
#         }
#         if kwargs["return_timestamps"] == "word":
#             response["words"] = [
#                 {"word": chunk["text"].strip(), "start": chunk["timestamp"][0], "end": chunk["timestamp"][1]}
#                 for chunk in chunks
#             ]
#         else:
#             response["segments"] = [
#                 {
#                     "id": i,
#                     # "seek": 0,
#                     "start": chunk["timestamp"][0],
#                     "end": chunk["timestamp"][1],
#                     "text": chunk["text"].strip(),
#                     # "tokens": [ ],
#                     # "temperature": 0.0,
#                     # "avg_logprob": -0.2860786020755768,
#                     # "compression_ratio": 1.2363636493682861,
#                     # "no_speech_prob": 0.00985979475080967
#                 }
#                 for i, chunk in enumerate(chunks)
#             ]
#
#         return JSONResponse(
#             content=response,
#             media_type="application/json",
#             headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"},
#         )
#
#     elif response_format == "srt":
#
#         def srt_time(t):
#             return "{:02d}:{:02d}:{:06.3f}".format(int(t // 3600), int(t // 60) % 60, t % 60).replace(".", ",")
#
#         return PlainTextResponse(
#             "\n".join(
#                 [
#                     f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
#                     for i, chunk in enumerate(result["chunks"], 1)
#                 ]
#             ),
#             media_type="text/srt; charset=utf-8",
#             headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"},
#         )
#
#     elif response_format == "vtt":
#
#         def vtt_time(t):
#             return "{:02d}:{:06.3f}".format(int(t // 60), t % 60)
#
#         return PlainTextResponse(
#             "\n".join(
#                 ["WEBVTT\n"]
#                 + [
#                     f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
#                     for chunk in result["chunks"]
#                 ]
#             ),
#             media_type="text/vtt; charset=utf-8",
#             headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"},
#         )
#
#
# @app.post("/v1/audio/transcriptions")
# async def transcriptions(
#     file: UploadFile,
#     model: str = Form(...),
#     language: Optional[str] = Form(None),
#     prompt: Optional[str] = Form(None),
#     response_format: Optional[str] = Form("json"),
#     temperature: Optional[float] = Form(None),
#     timestamp_granularities: List[str] = Form(["segment"]),
# ):
#     global pipe
#
#     kwargs = {"generate_kwargs": {"task": "transcribe"}}
#
#     if language:
#         kwargs["generate_kwargs"]["language"] = language
#     # May work soon, https://github.com/huggingface/transformers/issues/27317
#     #    if prompt:
#     #        kwargs["initial_prompt"] = prompt
#     if temperature:
#         kwargs["generate_kwargs"]["temperature"] = temperature
#         kwargs["generate_kwargs"]["do_sample"] = True
#
#     if response_format == "verbose_json" and "word" in timestamp_granularities:
#         kwargs["return_timestamps"] = "word"
#     else:
#         kwargs["return_timestamps"] = response_format in ["verbose_json", "srt", "vtt"]
#
#     return await whisper(file, response_format, **kwargs)
#
#
# @app.post("/v1/audio/translations")
# async def translations(
#     file: UploadFile,
#     model: str = Form(...),
#     prompt: Optional[str] = Form(None),
#     response_format: Optional[str] = Form("json"),
#     temperature: Optional[float] = Form(None),
# ):
#     global pipe
#
#     kwargs = {"generate_kwargs": {"task": "translate"}}
#
#     # May work soon, https://github.com/huggingface/transformers/issues/27317
#     #    if prompt:
#     #        kwargs["initial_prompt"] = prompt
#     if temperature:
#         kwargs["generate_kwargs"]["temperature"] = temperature
#         kwargs["generate_kwargs"]["do_sample"] = True
#
#     kwargs["return_timestamps"] = response_format in ["verbose_json", "srt", "vtt"]
#
#     return await whisper(file, response_format, **kwargs)
