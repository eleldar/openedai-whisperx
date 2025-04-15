import os
import subprocess
from pathlib import Path
from time import sleep
from types import NoneType

import pytest
import requests
from dotenv import load_dotenv
from Levenshtein import distance
from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose

load_dotenv("whisperx.env")


@pytest.mark.asyncio
class TestAPI:
    def setup_class(self):
        self._compose_file = "docker-compose.yml"
        subprocess.run(
            ["docker", "compose", "-f", self._compose_file, "up", "--build", "-d"],
            stdout=open(os.devnull, "w"),
            stderr=subprocess.STDOUT,
        )
        self._sleep = 1
        self._timeout = 10
        self._api_host = os.environ.get("TRANSCRIBATION_API_HOST")
        self._api_port = os.environ.get("TRANSCRIBATION_API_PORT")
        self._api_url = f"http://{self._api_host}:{self._api_port}"
        self._file_dir = Path("tests/datasets")
        self._model_name = "openai/whisper-small"
        self._model = OpenAI(
            base_url=f"{self._api_url}/v1",
            api_key="api_key",
        )
        self._distance_coef = 0.05
        connection = False
        timeout_counter = 0
        while not connection:
            try:
                requests.get(self._api_url)
                connection = True
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
                sleep(self._sleep)
                timeout_counter += 1
                if timeout_counter > self._timeout:
                    raise Exception("Setup timeout")

    def teardown_class(self):
        subprocess.run(
            ["docker", "compose", "-f", self._compose_file, "down"],
            stdout=open(os.devnull, "w"),
            stderr=subprocess.STDOUT,
        )
        connection = True
        timeout_counter = 0
        while connection:
            response = subprocess.run(["docker", "ps", "--filter"], stdout=subprocess.PIPE)
            response = str(response).find(f"{self._api_host}:{self._api_port}")
            if response < 0:
                connection = False
            else:
                sleep(self._sleep)
            if timeout_counter > self._timeout:
                raise Exception("Teardown timeout")
            timeout_counter += 1

    async def test_health(self):
        response = requests.get(f"{self._api_url}/health")
        assert 200 == response.status_code
        assert {"status": "ok"} == response.json()

    async def test_docs(self):
        response = requests.get(f"{self._api_url}/docs")
        assert 200 == response.status_code
        assert "FastAPI - Swagger UI" in response.text

    async def test_transcriptions_success(self):
        file_path = self._file_dir / "speech.wav"
        assert os.path.exists(file_path)
        assert isinstance(self._model, OpenAI)
        with open(file_path, "rb") as file:
            transcript = self._model.audio.transcriptions.create(
                model=self._model_name,
                file=file,
                response_format="verbose_json",
                timestamp_granularities="word",
                language="ru",
                temperature=0,
            )
        assert isinstance(transcript, TranscriptionVerbose)
        result = {
            "segments": list(map(dict, transcript.segments)),
            "word_segments": list(map(dict, transcript.word_segments)),
        }
        assert isinstance(result["segments"], list)
        for segment in result["segments"]:
            assert isinstance(segment["text"], str)
            assert isinstance(segment["start"], (float, NoneType))
            assert isinstance(segment["end"], (float, NoneType))
        for segment in result["word_segments"]:
            assert isinstance(segment["word"], str)
            assert isinstance(segment.get("start"), (float, NoneType))
            assert isinstance(segment.get("end"), (float, NoneType))
        with open(self._file_dir / "text.txt") as f:
            true_text = f.read().strip()
            pred_text = str(transcript.model_dump_json())
            threshold = len(true_text) * self._distance_coef
            assert distance(pred_text, true_text) <= threshold

    async def test_transcriptions_no_sound(self):
        file_path = self._file_dir / "no_sound.mp4"
        assert os.path.exists(file_path)
        assert isinstance(self._model, OpenAI)
        with open(file_path, "rb") as file:
            transcript = self._model.audio.transcriptions.create(
                model=self._model_name,
                file=file,
                response_format="verbose_json",
                timestamp_granularities="word",
                language="ru",
                temperature=0,
            )
        assert isinstance(transcript, TranscriptionVerbose)
        assert isinstance(transcript.duration, float)
        assert isinstance(transcript.language, str)
        assert isinstance(transcript.text, str)
        assert isinstance(transcript.segments, list)
        assert isinstance(transcript.words, list)
        assert isinstance(transcript.word_segments, list)
        assert transcript.duration == 0.0
        assert transcript.language == "ru"
        assert transcript.text == ""
        assert transcript.segments == []
        assert transcript.words == []
        assert transcript.word_segments == []
