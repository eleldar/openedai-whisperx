import argparse
import logging
import sys

import torch
import uvicorn

from config import state
from processor import app

logger = logging.getLogger(__name__)


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
    for _name, _model in state.model_mapping.items():
        app.register_model(name=_name, model=_model)
    args = parse_args(sys.argv[1:])
    state.model = args.model
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        compute_type = "float16" if torch.cuda.is_available() else "int8"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    uvicorn.run(
        app, host=args.host, port=args.port
    )  # , root_path=cwd, access_log=False, log_level="info", ssl_keyfile="cert.pem", ssl_certfile="cert.pem")
