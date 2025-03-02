import os
import re
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
        self._api_url = f'http://{os.environ.get("API_HOST")}:{os.environ.get("API_PORT")}'
        self._file_dir = Path("tests/datasets")
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
        api_host = os.environ.get("API_HOST")
        api_port = os.environ.get("API_PORT")
        subprocess.run(
            ["docker", "compose", "-f", self._compose_file, "down"],
            stdout=open(os.devnull, "w"),
            stderr=subprocess.STDOUT,
        )
        connection = True
        timeout_counter = 0
        while connection:
            response = subprocess.run(["docker", "ps", "--filter"], stdout=subprocess.PIPE)
            response = str(response).find(f"{api_host}:{api_port}")
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
                model="openai/whisper-small",
                file=file,
                response_format="verbose_json",
                timestamp_granularities="word",
                language="ru",
                temperature=0,
            )
        assert isinstance(transcript, TranscriptionVerbose)
        result = result = {
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
