"""Integration tests for ab_cli.commands.media."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ab_cli.commands import media


class TestMediaMain:
    def test_main_no_args_shows_help(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["media"])

        with pytest.raises(SystemExit) as exc_info:
            media.main()

        assert exc_info.value.code == 0
        assert "Audio and video utilities" in capsys.readouterr().out

    def test_extract_audio_invokes_ffmpeg(self, tmp_path, monkeypatch, capsys):
        input_file = tmp_path / "video.mp4"
        input_file.write_text("video")
        output_file = tmp_path / "audio.mp3"
        monkeypatch.setattr(sys, "argv", [
            "media",
            "extract-audio",
            str(input_file),
            "-o",
            str(output_file),
            "-y",
        ])
        monkeypatch.setattr(media, "require_tool", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(media, "ensure_audio_stream", lambda path: None)

        with patch("ab_cli.commands.media.run_command", return_value=MagicMock(returncode=0)) as mock_run:
            media.main()

        cmd = mock_run.call_args.args[0]
        assert cmd[:2] == ["ffmpeg", "-y"]
        assert str(input_file) in cmd
        assert str(output_file) in cmd
        assert str(output_file) in capsys.readouterr().out

    def test_extract_audio_refuses_existing_output(self, tmp_path, monkeypatch, capsys):
        input_file = tmp_path / "video.mp4"
        input_file.write_text("video")
        output_file = tmp_path / "audio.mp3"
        output_file.write_text("exists")
        monkeypatch.setattr(sys, "argv", [
            "media",
            "extract-audio",
            str(input_file),
            "-o",
            str(output_file),
        ])

        with pytest.raises(SystemExit) as exc_info:
            media.main()

        assert exc_info.value.code == 1
        assert "already exists" in capsys.readouterr().err

    def test_transcribe_audio_segments_and_writes_output(self, tmp_path, monkeypatch, mock_config, mock_env, capsys):
        input_file = tmp_path / "audio.mp3"
        input_file.write_bytes(b"audio")
        output_file = tmp_path / "transcript.txt"
        chunk1 = tmp_path / "chunk1.mp3"
        chunk2 = tmp_path / "chunk2.mp3"
        chunk1.write_bytes(b"one")
        chunk2.write_bytes(b"two")
        monkeypatch.setattr(sys, "argv", [
            "media",
            "transcribe",
            str(input_file),
            "-o",
            str(output_file),
            "--model",
            "openai/gpt-4o-mini-transcribe",
            "--language",
            "pt",
            "--chunk-seconds",
            "10",
            "--temperature",
            "0.2",
        ])

        with patch("ab_cli.commands.media.segment_media", return_value=[chunk1, chunk2]) as mock_segment:
            with patch("ab_cli.commands.media.transcribe_audio_openrouter") as mock_transcribe:
                mock_transcribe.side_effect = [
                    {"text": "primeiro"},
                    {"text": "segundo"},
                ]
                media.main()

        assert output_file.read_text(encoding="utf-8") == "primeiro\n\nsegundo\n"
        assert mock_segment.call_args.args[2] == 10
        first_call = mock_transcribe.call_args_list[0].kwargs
        assert first_call["language"] == "pt"
        assert first_call["temperature"] == 0.2
        assert str(output_file) in capsys.readouterr().out

    def test_transcribe_video_uses_segment_media(self, tmp_path, monkeypatch, mock_config, mock_env, capsys):
        input_file = tmp_path / "video.mp4"
        input_file.write_bytes(b"video")
        chunk = tmp_path / "chunk.mp3"
        chunk.write_bytes(b"audio")
        monkeypatch.setattr(sys, "argv", ["media", "transcribe", str(input_file)])

        with patch("ab_cli.commands.media.segment_media", return_value=[chunk]) as mock_segment:
            with patch("ab_cli.commands.media.transcribe_audio_openrouter", return_value={"text": "texto"}):
                media.main()

        assert mock_segment.call_args.args[0] == input_file
        assert "texto" in capsys.readouterr().out

    def test_transcribe_api_failure_exits_1(self, tmp_path, monkeypatch, mock_config, mock_env, capsys):
        input_file = tmp_path / "audio.mp3"
        input_file.write_bytes(b"audio")
        chunk = tmp_path / "chunk.mp3"
        chunk.write_bytes(b"audio")
        monkeypatch.setattr(sys, "argv", ["media", "transcribe", str(input_file)])

        with patch("ab_cli.commands.media.segment_media", return_value=[chunk]):
            with patch("ab_cli.commands.media.transcribe_audio_openrouter", return_value=None):
                with pytest.raises(SystemExit) as exc_info:
                    media.main()

        assert exc_info.value.code == 1
        assert "chunk 1" in capsys.readouterr().err
