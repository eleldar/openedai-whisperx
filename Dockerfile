FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
	        ffmpeg python-is-python3 python3-pip \
			libcudnn8 libcudnn8-dev libcudnn8-samples

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY *.py /app/
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
EXPOSE ${TRANSCRIBATION_API_PORT}
CMD uvicorn main:app --host ${TRANSCRIBATION_API_HOST} --port ${TRANSCRIBATION_API_PORT}
