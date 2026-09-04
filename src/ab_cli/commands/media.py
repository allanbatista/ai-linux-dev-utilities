#!/usr/bin/env python3
"""Audio and video utilities."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ab_cli.core.config import get_config
from ab_cli.utils.api import transcribe_audio_openrouter
from ab_cli.utils.error_handling import handle_cli_errors

SUPPORTED_FORMATS = ("mp3", "wav", "m4a", "flac")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
DEFAULT_TRANSCRIPTION_MODEL = "openai/gpt-4o-mini-transcribe"
DEFAULT_DIARIZATION_MODEL = "x-ai/grok-stt-1.0"
DEFAULT_CHUNK_SECONDS = 300
# Diarization speaker IDs reset per request; prefer one large chunk when possible.
DEFAULT_DIARIZE_CHUNK_SECONDS = 3600


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


def format_diarized_transcript(words: List[Dict[str, Any]]) -> str:
    """Group consecutive words by speaker into plain-text lines.

    Labels are generic speaker IDs (Speaker 0, Speaker 1, ...), not person names.
    """
    if not words:
        return ""

    lines: List[str] = []
    current_speaker: Optional[Any] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_speaker
        if not buffer:
            return
        label = f"Speaker {current_speaker}" if current_speaker is not None else "Speaker ?"
        lines.append(f"{label}: {' '.join(buffer).strip()}")
        buffer = []

    for word in words:
        text = str(word.get("text") or word.get("word") or "").strip()
        if not text:
            continue
        speaker = word.get("speaker")
        if current_speaker is None:
            current_speaker = speaker
        elif speaker is not None and speaker != current_speaker:
            flush()
            current_speaker = speaker
        buffer.append(text)

    flush()
    return "\n".join(lines)


def _words_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = result.get("words") or []
    if words:
        return list(words)
    # Some providers put speaker labels on segments instead of words.
    segments = result.get("segments") or []
    out: List[Dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append({"text": text, "speaker": seg.get("speaker")})
    return out


def transcribe_chunks(
    chunks: List[Path],
    model: str,
    language: Optional[str],
    temperature: Optional[float],
    diarize: bool = False,
) -> str:
    config = get_config()
    settings = config.get_api_settings()
    texts: List[str] = []
    diarized_parts: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        result = transcribe_audio_openrouter(
            str(chunk),
            "mp3",
            model,
            settings["timeout_seconds"],
            language=language,
            temperature=temperature,
            diarize=diarize,
            api_key_env=settings["api_key_env"],
            api_base=settings["api_base"],
        )
        if result is None:
            raise MediaError(f"Transcription failed for chunk {index}.")

        if diarize:
            words = _words_from_result(result)
            if not words:
                raise MediaError(
                    "Diarization requested but response has no speaker labels "
                    "(words/segments with speaker). Use model x-ai/grok-stt-1.0 "
                    "or omit --diarize for plain transcription."
                )
            part = format_diarized_transcript(words)
            if part:
                diarized_parts.append(part)
        else:
            text = (result.get("text") or "").strip()
            if text:
                texts.append(text)

    if diarize:
        # Speaker IDs reset per API chunk; join parts without inventing IDs.
        return "\n\n".join(diarized_parts)
    return "\n\n".join(texts)


def cmd_extract_audio(args) -> None:
    input_path = validate_input(args.input)
    output_path = resolve_output_path(input_path, args.output, args.format, args.yes)
    extract_audio_file(input_path, output_path, args.format, args.yes)
    print(output_path)


def cmd_transcribe(args) -> None:
    input_path = validate_input(args.input)
    config = get_config()
    diarize = bool(args.diarize)
    if args.model:
        model = args.model
    elif diarize:
        model = config.get("commands.media.diarization_model", DEFAULT_DIARIZATION_MODEL)
    else:
        model = config.get("commands.media.transcription_model", DEFAULT_TRANSCRIPTION_MODEL)

    if args.chunk_seconds is not None:
        chunk_seconds = args.chunk_seconds
    elif diarize:
        chunk_seconds = config.get(
            "commands.media.diarize_chunk_seconds",
            DEFAULT_DIARIZE_CHUNK_SECONDS,
        )
    else:
        chunk_seconds = config.get("commands.media.chunk_seconds", DEFAULT_CHUNK_SECONDS)

    temperature = args.temperature
    if temperature is None:
        temperature = config.get("commands.media.temperature", 0)

    with tempfile.TemporaryDirectory(prefix="ab-media-") as temp_dir:
        chunks = segment_media(input_path, Path(temp_dir), int(chunk_seconds))
        transcript = transcribe_chunks(
            chunks, model, args.language, temperature, diarize=diarize
        )

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
  ab media transcribe reuniao.mp3 --diarize
  ab media transcribe call.wav --diarize --model x-ai/grok-stt-1.0
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    extract = subparsers.add_parser("extract-audio", help="Extract audio from a video")
    extract.add_argument("input", help="Input video/audio file")
    extract.add_argument("-o", "--output", help="Output audio path")
    extract.add_argument("--format", choices=SUPPORTED_FORMATS, default="mp3", help="Output format")
    extract.add_argument("-y", "--yes", action="store_true", help="Overwrite output if it exists")

    transcribe = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio or video to plain text (mp4/avi/webm/mp3/wav/...)",
    )
    transcribe.add_argument("input", help="Input audio/video file")
    transcribe.add_argument("-o", "--output", help="Output transcript path")
    transcribe.add_argument("--model", help="OpenRouter STT model")
    transcribe.add_argument("--language", help="ISO-639-1 language code")
    transcribe.add_argument("--chunk-seconds", type=int, help="Audio chunk length in seconds")
    transcribe.add_argument("--temperature", type=float, help="Transcription sampling temperature")
    transcribe.add_argument(
        "--diarize",
        action="store_true",
        help=(
            "Label speakers (Speaker 0, Speaker 1, ...). "
            "Default model: x-ai/grok-stt-1.0. Labels are not real names."
        ),
    )

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
