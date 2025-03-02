from pathlib import Path

import torch
from pydantic import BaseModel, Field


class Config(BaseModel):
    model_mapping: dict[str, str] = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large-v1",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v3-turbo": "large-v3-turbo",
    }
    model: str = Field("openai/whisper-large-v3")
    batch_size: int = Field(16)
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    compute_type: str = Field(default_factory=lambda: "float16" if torch.cuda.is_available() else "int8")
    temperature: float = Field(0.0)
    language: str = Field("ru")
    prompt: str | None = Field(None)
    response_format: str = Field("verbose_json")
    timestamp_granularities: str = "word"
    tempfiles: Path = Path("/app/tempfiles")
    model_dir: Path = Path("/app/models")


state = Config()
