OpenedAI WhisperX
----------------

Notice: This software is forked from https://github.com/matatonic/openedai-whisper/blob/main/whisper.py

----

An OpenAI API compatible speech to text server for audio transcription and translations.

- Compatible with the OpenAI audio/transcriptions and audio/translations API
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

API Compatibility:
- [X] /v1/audio/transcriptions
- [ ] /v1/audio/translations

Parameter Support:
- [X] `file`
- [X] `model`
- [X] `language`
- [X] `prompt`
- [X] `temperature`
- [ ] `response_format`:
- - [ ] `json`
- - [ ] `text`
- - [ ] `srt`
- - [ ] `vtt`
- - [ ] `verbose_json` *(always return verbose json)

Details:
* CUDA or CPU support (automatically detected)
* float16 or int8 support (automatically detected)

Tested whisper models:
* openai/whisper-large-v3 (the default)
* openai/whisper-small
* ...

Version: 0.1.0, Last update: 2025-03-03


API Documentation
-----------------

## Usage

* [OpenAI Speech to text guide](https://platform.openai.com/docs/guides/speech-to-text)
* [OpenAI API Transcription Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)
* [OpenAI API Translation Reference](https://platform.openai.com/docs/api-reference/audio/createTranslation)


Docker support
--------------

```shell
docker compose --env-file whisperx.env up --build
```


Installation instructions
-------------------------

You will need to install CUDA for your operating system if you want to use CUDA.

```shell
# install ffmpeg
sudo apt install ffmpeg libcudnn8 libcudnn8-dev libcudnn8-samples
# Install the Python requirements
pip install -r requirements.txt
```


CLI Usage
---------

```
Usage: whisper.py [-m <model_name>] [-d <device>] [-t <dtype>] [-P <port>] [-H <host>] [--preload]


Description:
OpenedAI Whisper API Server

Options:
-h, --help            Show this help message and exit.
-m MODEL, --model MODEL
                      The model to use for transcription.
                      Ex. openai/whisper-small (default: openai/whisper-large-v3)
-d DEVICE, --device DEVICE
                      Set the torch device for the model. Ex. cuda:1 (default: auto)
-t DTYPE, --dtype DTYPE
                      Set the torch data type for processing (float16, int8) (default: auto)
-P PORT, --port PORT  Server tcp port (default: 8000)
-H HOST, --host HOST  Host to listen on, Ex. 0.0.0.0 (default: localhost)
--preload             Preload model and exit. (default: False)
```


Sample API Usage
----------------

You can use it like this:

```shell
curl -s http://localhost:8000/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F model="openai/whisper-large-v3" -F file="@audio.mp3" -F response_format=text
```

Or just like this:

```shell
curl -s http://localhost:8000/v1/audio/transcriptions -F model="openai/whisper-large-v3" -F file="@audio.mp3"
```

Or like this example from the [OpenAI Speech to text guide Quickstart](https://platform.openai.com/docs/guides/speech-to-text/quickstart):

```python
from openai import OpenAI
client = OpenAI(api_key='sk-1111', base_url='http://localhost:8000/v1')

audio_file = open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(model="openai/whisper-large-v3", file=audio_file)
print(transcription.text)
```
