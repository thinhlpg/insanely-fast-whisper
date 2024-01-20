import json
import argparse
from transformers import pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import torch

from .utils.diarization_pipeline import diarize
from .utils.result import build_result

parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=True,
    type=str,
    help="Path or URL to the audio file to be transcribed.",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")',
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf_token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)

parser.add_argument(
    "--chunk_size",
    required=False,
    default=30,
    type=int,
    help="Chunk size for Whisper ASR. (default: 30)",
)

parser.add_argument(
    "--stride",
    required=False,
    default=None,
    type=int,
    help="Stride for Whisper ASR. (default: chunk_size/6)",
)

parser.add_argument(
    "--verbose",
    required=False,
    action="store_true",
    help="Prints out the arguments passed to the script. (default: False)",
)


def main():
    args = parser.parse_args()

    # print args
    if args.verbose:
        print("===========🤗 Arguments:============")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs={"attn_implementation": "flash_attention_2"} if args.flash else {"attn_implementation": "sdpa"},
    )

    if args.device_id == "mps":
        torch.mps.empty_cache()
    # elif not args.flash:
    # pipe.model = pipe.model.to_bettertransformer()

    ts = "word" if args.timestamp == "word" else True

    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task, "language": language}

    if ts == "word":
        generate_kwargs["return_segments"] = True

    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        outputs = pipe(
            args.file_name,
            chunk_length_s=args.chunk_size,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
            stride_length_s=[args.stride, args.stride] if args.stride else None,
        )

    if args.hf_token != "no_token":
        speakers_transcript = diarize(args, outputs)
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(f"Voila!✨ Your file has been transcribed go check it out over here 👉 {args.transcript_path}")
