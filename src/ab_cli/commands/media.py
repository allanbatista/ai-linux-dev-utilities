#!/usr/bin/env python3
"""Audio and video utilities."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from ab_cli.core.config import get_config
from ab_cli.utils.api import transcribe_audio_openrouter
from ab_cli.utils.error_handling import handle_cli_errors

SUPPORTED_FORMATS = ("mp3", "wav", "m4a", "flac")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
DEFAULT_TRANSCRIPTION_MODEL = "openai/gpt-4o-mini-transcribe"
DEFAULT_CHUNK_SECONDS = 300


class MediaError(Exception):
    """Raised for media command failures."""


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"Required dependency not found: {name}")
    return path


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise MediaError(detail or f"Command failed: {' '.join(cmd)}")
    return result


def validate_input(path: str) -> Path:
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise MediaError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise MediaError(f"Input path is not a file: {input_path}")
    return input_path


def has_audio_stream(input_path: Path) -> bool:
    require_tool("ffprobe")
    result = subprocess.run([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(input_path),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return "audio" in result.stdout.lower()


def ensure_audio_stream(input_path: Path) -> None:
    if not has_audio_stream(input_path):
        raise MediaError(f"No audio stream found in: {input_path}")


def resolve_output_path(input_path: Path, output: Optional[str], fmt: str, overwrite: bool) -> Path:
    output_path = Path(output).expanduser() if output else input_path.with_suffix(f".{fmt}")
    if output_path.exists() and not overwrite:
        raise MediaError(f"Output file already exists: {output_path}. Use -y to overwrite.")
    return output_path


def codec_args(fmt: str) -> List[str]:
    return {
        "mp3": ["-codec:a", "libmp3lame", "-q:a", "2"],
        "wav": ["-codec:a", "pcm_s16le"],
        "m4a": ["-codec:a", "aac"],
        "flac": ["-codec:a", "flac"],
    }[fmt]


def extract_audio_file(input_path: Path, output_path: Path, fmt: str, overwrite: bool) -> Path:
    require_tool("ffmpeg")
    ensure_audio_stream(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        *codec_args(fmt),
        str(output_path),
    ])
    return output_path


def segment_media(input_path: Path, workdir: Path, chunk_seconds: int) -> List[Path]:
    require_tool("ffmpeg")
    ensure_audio_stream(input_path)
    pattern = workdir / "chunk_%05d.mp3"
    run_command([
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-codec:a",
        "libmp3lame",
        str(pattern),
    ])
    chunks = sorted(workdir.glob("chunk_*.mp3"))
    if not chunks:
        raise MediaError("No audio chunks were generated.")
    return chunks


def transcribe_chunks(
    chunks: List[Path],
    model: str,
    language: Optional[str],
    temperature: Optional[float],
) -> str:
    config = get_config()
    settings = config.get_api_settings()
    texts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        result = transcribe_audio_openrouter(
            str(chunk),
            "mp3",
            model,
            settings["timeout_seconds"],
            language=language,
            temperature=temperature,
            api_key_env=settings["api_key_env"],
            api_base=settings["api_base"],
        )
        if result is None:
            raise MediaError(f"Transcription failed for chunk {index}.")
        text = (result.get("text") or "").strip()
        if text:
            texts.append(text)

    return "\n\n".join(texts)


def cmd_extract_audio(args) -> None:
    input_path = validate_input(args.input)
    output_path = resolve_output_path(input_path, args.output, args.format, args.yes)
    extract_audio_file(input_path, output_path, args.format, args.yes)
    print(output_path)


def cmd_transcribe(args) -> None:
    input_path = validate_input(args.input)
    config = get_config()
    model = args.model or config.get("commands.media.transcription_model", DEFAULT_TRANSCRIPTION_MODEL)
    chunk_seconds = args.chunk_seconds or config.get("commands.media.chunk_seconds", DEFAULT_CHUNK_SECONDS)
    temperature = args.temperature
    if temperature is None:
        temperature = config.get("commands.media.temperature", 0)

    with tempfile.TemporaryDirectory(prefix="ab-media-") as temp_dir:
        chunks = segment_media(input_path, Path(temp_dir), int(chunk_seconds))
        transcript = transcribe_chunks(chunks, model, args.language, temperature)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript + ("\n" if transcript else ""), encoding="utf-8")
        print(output_path)
    else:
        print(transcript)


@handle_cli_errors
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio and video utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ab media extract-audio video.mp4
  ab media extract-audio video.mp4 -o audio.wav --format wav
  ab media transcribe audio.mp3
  ab media transcribe video.mp4 -o transcript.txt --language pt
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    extract = subparsers.add_parser("extract-audio", help="Extract audio from a video")
    extract.add_argument("input", help="Input video/audio file")
    extract.add_argument("-o", "--output", help="Output audio path")
    extract.add_argument("--format", choices=SUPPORTED_FORMATS, default="mp3", help="Output format")
    extract.add_argument("-y", "--yes", action="store_true", help="Overwrite output if it exists")

    transcribe = subparsers.add_parser("transcribe", help="Transcribe audio or video to text")
    transcribe.add_argument("input", help="Input audio/video file")
    transcribe.add_argument("-o", "--output", help="Output transcript path")
    transcribe.add_argument("--model", help="OpenRouter STT model")
    transcribe.add_argument("--language", help="ISO-639-1 language code")
    transcribe.add_argument("--chunk-seconds", type=int, help="Audio chunk length in seconds")
    transcribe.add_argument("--temperature", type=float, help="Transcription sampling temperature")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "extract-audio":
            cmd_extract_audio(args)
        elif args.command == "transcribe":
            cmd_transcribe(args)
    except MediaError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
