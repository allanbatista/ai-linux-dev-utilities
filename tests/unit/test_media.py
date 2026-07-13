"""Unit tests for media utilities."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from ab_cli.commands import media
from ab_cli.utils.api import transcribe_audio_openrouter


class TestMediaHelpers:
    def test_require_tool_missing(self, monkeypatch):
        monkeypatch.setattr(media.shutil, "which", lambda name: None)

        with pytest.raises(media.MediaError, match="Required dependency not found"):
            media.require_tool("ffmpeg")

    def test_resolve_output_refuses_overwrite(self, tmp_path):
        source = tmp_path / "video.mp4"
        source.write_text("video")
        output = tmp_path / "audio.mp3"
        output.write_text("exists")

        with pytest.raises(media.MediaError, match="already exists"):
            media.resolve_output_path(source, str(output), "mp3", False)

    def test_resolve_output_allows_overwrite(self, tmp_path):
        source = tmp_path / "video.mp4"
        source.write_text("video")
        output = tmp_path / "audio.mp3"
        output.write_text("exists")

        result = media.resolve_output_path(source, str(output), "mp3", True)

        assert result == output

    def test_codec_args_mp3(self):
        assert media.codec_args("mp3") == ["-codec:a", "libmp3lame", "-q:a", "2"]

    def test_has_audio_stream_false_on_ffprobe_failure(self, tmp_path, monkeypatch):
        input_file = tmp_path / "video.mp4"
        input_file.write_text("media")
        monkeypatch.setattr(media, "require_tool", lambda name: f"/usr/bin/{name}")

        failed = MagicMock(returncode=1, stdout="", stderr="no stream")
        with patch("subprocess.run", return_value=failed):
            assert media.has_audio_stream(input_file) is False

    def test_segment_media_generates_expected_ffmpeg_command(self, tmp_path, monkeypatch):
        input_file = tmp_path / "input.mp4"
        input_file.write_text("media")
        chunk = tmp_path / "chunk_00000.mp3"

        monkeypatch.setattr(media, "require_tool", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(media, "ensure_audio_stream", lambda path: None)

        def fake_run(cmd):
            chunk.write_bytes(b"audio")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("ab_cli.commands.media.run_command", side_effect=fake_run) as mock_run:
            chunks = media.segment_media(input_file, tmp_path, 123)

        assert chunks == [chunk]
        cmd = mock_run.call_args.args[0]
        assert "-segment_time" in cmd
        assert "123" in cmd
        assert "-ar" in cmd
        assert "16000" in cmd


class TestOpenRouterTranscription:
    def test_transcribe_audio_payload(self, tmp_path, monkeypatch):
        audio = tmp_path / "chunk.mp3"
        audio.write_bytes(b"abc")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        response = MagicMock()
        response.json.return_value = {
            "text": "hello",
            "usage": {"seconds": 1.2},
        }

        with patch("requests.post", return_value=response) as mock_post:
            result = transcribe_audio_openrouter(
                str(audio),
                "mp3",
                "openai/gpt-4o-mini-transcribe",
                30,
                language="en",
                temperature=0.1,
            )

        assert result["text"] == "hello"
        url = mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        assert url == "https://openrouter.ai/api/v1/audio/transcriptions"
        assert payload["input_audio"]["data"] == "YWJj"
        assert payload["input_audio"]["format"] == "mp3"
        assert payload["model"] == "openai/gpt-4o-mini-transcribe"
        assert payload["language"] == "en"
        assert payload["temperature"] == 0.1

    def test_transcribe_audio_missing_api_key(self, tmp_path, monkeypatch, capsys):
        audio = tmp_path / "chunk.mp3"
        audio.write_bytes(b"abc")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = transcribe_audio_openrouter(str(audio), "mp3", "model", 30)

        assert result is None
        assert "OPENROUTER_API_KEY" in capsys.readouterr().err

    def test_transcribe_audio_http_error(self, tmp_path, monkeypatch, capsys):
        audio = tmp_path / "chunk.mp3"
        audio.write_bytes(b"abc")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        response = MagicMock()
        response.text = "rate limited"
        error = requests.exceptions.HTTPError("429")
        error.response = response
        response.raise_for_status.side_effect = error

        with patch("requests.post", return_value=response):
            result = transcribe_audio_openrouter(str(audio), "mp3", "model", 30)

        assert result is None
        assert "rate limited" in capsys.readouterr().err
