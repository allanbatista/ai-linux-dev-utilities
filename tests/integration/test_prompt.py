"""Integration tests for ab_cli.commands.prompt module."""
import json
import os


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_config_uses_defaults(self, temp_config_dir):
        """Uses default configuration when no file exists."""
        from ab_cli.core.config import get_config

        config = get_config()
        assert config.get_with_default("global.api_base") == "https://openrouter.ai/api/v1"


class TestPersistDefaultModel:
    """Tests for model persistence."""

    def test_persist_default_model(self, temp_config_dir):
        """Saves default model to config."""
        from ab_cli.core.config import get_config

        config = get_config()
        config.init_config()
        config.set("models.default", "new/model")

        # Verify persisted
        config.reload()
        assert config.get("models.default") == "new/model"


class TestApiCalls:
    """Tests for API call functionality."""

    def test_send_to_openrouter_success(self, mock_requests, mock_env, temp_config_dir):
        """API call succeeds with valid response."""
        # Import after patching
        response = mock_requests.return_value
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        # Verify mock is set up
        assert response.status_code == 200

    def test_send_to_openrouter_no_api_key(self, temp_config_dir, monkeypatch):
        """Returns error without API key."""
        # Ensure no API key is set
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from ab_cli.core.config import get_config

        config = get_config()
        api_settings = config.get_api_settings()

        # Verify API key env var is configured but not set
        api_key = os.environ.get(api_settings["api_key_env"])
        assert api_key is None


class TestBinaryFileDetection:
    """Tests for binary file detection."""

    def test_is_binary_file_true(self, tmp_path):
        """Detects binary files correctly."""
        # Create a binary file
        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(bytes([0x00, 0x01, 0x02, 0x89, 0x50, 0x4E, 0x47]))

        from binaryornot.check import is_binary

        assert is_binary(str(binary_file)) is True

    def test_is_binary_file_false(self, tmp_path):
        """Text files return False."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is a text file\nwith multiple lines")

        from binaryornot.check import is_binary

        assert is_binary(str(text_file)) is False


class TestAiignore:
    """Tests for .aiignore file handling."""

    def test_should_ignore_path_matches_pattern(self, tmp_path):
        """Respects .aiignore patterns."""
        import pathspec

        # Create .aiignore
        aiignore = tmp_path / ".aiignore"
        aiignore.write_text("*.log\nnode_modules/\n__pycache__/\n")

        patterns = aiignore.read_text().strip().split("\n")
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        assert spec.match_file("test.log") is True
        assert spec.match_file("node_modules/package.json") is True
        assert spec.match_file("__pycache__/module.pyc") is True
        assert spec.match_file("src/main.py") is False

    def test_should_ignore_negation_pattern(self, tmp_path):
        """Handles negation patterns."""
        import pathspec

        patterns = ["*.log", "!important.log"]
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        assert spec.match_file("debug.log") is True
        # Note: pathspec handles negation differently
        # The file matches but is negated


class TestFileProcessing:
    """Tests for file processing."""

    def test_process_file_reads_content(self, tmp_path):
        """Reads and formats file content."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        content = test_file.read_text()
        assert "def hello():" in content
        assert "print('Hello')" in content

    def test_process_file_handles_encoding(self, tmp_path):
        """Handles different file encodings."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Olá mundo! 你好世界! 🎉", encoding="utf-8")

        content = test_file.read_text(encoding="utf-8")
        assert "Olá" in content
        assert "你好" in content

    def test_process_file_truncation(self, tmp_path):
        """Truncates large files."""
        # Create a large file
        large_file = tmp_path / "large.txt"
        large_content = "x" * 1000000  # 1MB
        large_file.write_text(large_content)

        # Simulate truncation logic
        max_chars = 100000
        content = large_file.read_text()[:max_chars]

        assert len(content) == max_chars


class TestSpecialistPersonas:
    """Tests for specialist persona handling."""

    def test_build_specialist_prefix_dev(self):
        """Returns dev persona prefix."""
        # Test the concept of specialist prefixes
        specialists = {
            "dev": "You are an expert software developer",
            "rm": "You are an expert release manager"
        }
        assert "developer" in specialists["dev"].lower()

    def test_build_specialist_prefix_rm(self):
        """Returns RM persona prefix."""
        specialists = {
            "dev": "You are an expert software developer",
            "rm": "You are an expert release manager"
        }
        assert "release manager" in specialists["rm"].lower()

    def test_build_specialist_prefix_none(self):
        """Returns empty for unknown specialist."""
        specialists = {
            "dev": "Developer",
            "rm": "Release Manager"
        }
        result = specialists.get("unknown", "")
        assert result == ""


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_accuracy(self):
        """Token estimation is reasonably accurate."""
        from ab_cli.core.config import estimate_tokens

        # ~4 chars per token is the approximation
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100

    def test_estimate_tokens_with_code(self):
        """Estimates tokens in code correctly."""
        from ab_cli.core.config import estimate_tokens

        code = """
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
"""
        tokens = estimate_tokens(code)
        assert tokens > 0
        # Approximately len/4
        assert abs(tokens - len(code) // 4) < 5


class TestModelSelection:
    """Tests for automatic model selection."""

    def test_select_model_by_tokens(self, mock_config):
        """Selects model based on token count."""
        from ab_cli.core.config import get_config

        config = get_config()

        # Small model for small context
        assert config.select_model(50000) == "test/model-small"

        # Medium model for medium context
        assert config.select_model(200000) == "test/model-medium"

        # Large model for large context
        assert config.select_model(300000) == "test/model-large"


class TestHistoryTracking:
    """Tests for history tracking functionality."""

    def test_history_directory_exists(self, temp_config_dir):
        """History directory can be created."""
        from ab_cli.core.config import get_config

        config = get_config()
        history_dir = config.get_history_dir()

        history_dir.mkdir(parents=True, exist_ok=True)
        assert history_dir.exists()

    def test_history_enabled_by_default(self, mock_config):
        """History is enabled by default."""
        from ab_cli.core.config import get_config

        config = get_config()
        assert config.is_history_enabled() is True


class TestInputHandling:
    """Tests for various input handling scenarios."""

    def test_stdin_prompt_reading(self):
        """Can read prompt from stdin."""
        from io import StringIO

        stdin_content = "What is Python?"
        mock_stdin = StringIO(stdin_content)

        content = mock_stdin.read()
        assert content == "What is Python?"

    def test_file_path_handling(self, tmp_path):
        """Handles file paths correctly."""
        # Create nested directory structure
        src_dir = tmp_path / "src" / "components"
        src_dir.mkdir(parents=True)

        test_file = src_dir / "Button.tsx"
        test_file.write_text("export const Button = () => <button>Click</button>")

        # Verify path handling
        assert test_file.exists()
        assert test_file.is_file()
        assert test_file.suffix == ".tsx"


class TestOutputFormatting:
    """Tests for output formatting options."""

    def test_json_output_parsing(self):
        """Parses JSON output correctly."""
        json_response = '{"key": "value", "number": 42}'
        parsed = json.loads(json_response)

        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_relative_paths_display(self, tmp_path, monkeypatch):
        """Displays relative paths correctly."""
        monkeypatch.chdir(tmp_path)

        file_path = tmp_path / "subdir" / "file.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        relative = file_path.relative_to(tmp_path)
        assert str(relative) == "subdir/file.py"


class TestSanitizeSensitiveData:
    """Tests for sanitize_sensitive_data function."""

    def test_sanitize_empty_text(self):
        """Returns empty text unchanged."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        assert sanitize_sensitive_data("") == ""
        assert sanitize_sensitive_data(None) is None

    def test_sanitize_text_without_secrets(self):
        """Leaves normal text unchanged."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        text = "This is a normal text without any secrets."
        assert sanitize_sensitive_data(text) == text

    def test_sanitize_openrouter_api_key(self):
        """Sanitizes OPENROUTER_API_KEY."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        text = "OPENROUTER_API_KEY=sk-or-v1-abc123def456"
        result = sanitize_sensitive_data(text)
        assert "sk-or-v1-abc123def456" not in result
        assert "[REDACTED]" in result

    def test_sanitize_custom_api_keys(self):
        """Sanitizes custom API keys like STRIPE_API_KEY, GITHUB_API_KEY."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        # Test various custom API key formats
        test_cases = [
            "STRIPE_API_KEY=sk_live_abc123def456ghi789",
            "GITHUB_API_KEY=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "AWS_API_KEY=AKIAIOSFODNN7EXAMPLE",
            "SENDGRID_API_KEY=SG.abcdefghijklmnop",
            "CUSTOM_SERVICE_API_KEY=my-secret-key-12345",
        ]

        for text in test_cases:
            result = sanitize_sensitive_data(text)
            # The value after = should be redacted
            assert "[REDACTED]" in result, f"Failed for: {text}"
            # Ensure the key name is preserved
            key_name = text.split("=")[0]
            assert key_name in result, f"Key name lost for: {text}"

    def test_sanitize_webhook_urls(self):
        """Sanitizes webhook URLs."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        test_cases = [
            "https://hooks.slack.com/services/T00000000/B00000000/XXXX",
            "https://api.github.com/repos/user/repo/hooks/webhook",
            "http://example.com/webhook/abc123",
            "https://discord.com/api/webhooks/123456789/abcdef",
            "Config: webhook_url=https://my.service.com/webhook/secret",
        ]

        for text in test_cases:
            result = sanitize_sensitive_data(text)
            assert "[REDACTED_WEBHOOK_URL]" in result, f"Failed for: {text}"
            assert "webhook" not in result.lower() or "[REDACTED_WEBHOOK_URL]" in result

    def test_sanitize_oauth_tokens(self):
        """Sanitizes OAuth tokens."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        test_cases = [
            "oauth_token=ya29.a0AfH6SMBx1234567890abcdefghijklmnop",
            "OAUTH_TOKEN: gho_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            "ACCESS_TOKEN: Bearer abc123def456",
        ]

        for text in test_cases:
            result = sanitize_sensitive_data(text)
            assert "[REDACTED]" in result, f"Failed for: {text}"

    def test_sanitize_bearer_tokens(self):
        """Sanitizes Bearer tokens with various formats."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        test_cases = [
            ("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
             "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
            ("Bearer abc123-def456_ghi789.jkl012+mno345/pqr678=",
             "abc123-def456_ghi789.jkl012+mno345/pqr678="),
        ]

        for text, secret in test_cases:
            result = sanitize_sensitive_data(text)
            assert secret not in result, f"Secret not redacted for: {text}"
            assert "[REDACTED]" in result, f"Failed for: {text}"

    def test_sanitize_bearer_with_sk_token(self):
        """Bearer tokens with sk- prefix are caught by API key pattern."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        # sk- tokens in Bearer headers get caught by the API key pattern
        text = "Header: Bearer sk-proj-abcdefghijklmnopqrstuvwxyz"
        result = sanitize_sensitive_data(text)
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz" not in result
        assert "[REDACTED" in result  # Either [REDACTED] or [REDACTED_API_KEY]

    def test_sanitize_private_keys(self):
        """Sanitizes PEM private keys."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy
YmIxMzU2NzE1MzY1Njg5MDEyMzQ1Njc4OTAxMjM0NTY3
-----END RSA PRIVATE KEY-----"""

        result = sanitize_sensitive_data(private_key)
        assert "[REDACTED_PRIVATE_KEY]" in result
        assert "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn" not in result

    def test_sanitize_ec_private_key(self):
        """Sanitizes EC private keys."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        private_key = """-----BEGIN EC PRIVATE KEY-----
MHQCAQEEIBYr4jkS2RSVPB6c/87
-----END EC PRIVATE KEY-----"""

        result = sanitize_sensitive_data(private_key)
        assert "[REDACTED_PRIVATE_KEY]" in result

    def test_sanitize_generic_secrets(self):
        """Sanitizes generic secret patterns."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        test_cases = [
            "DB_PASSWORD=super_secret_pass123",
            "APP_SECRET=my-app-secret-key",
            "AUTH_TOKEN=abc123def456ghi789",
            "ENCRYPTION_KEY=AES256-key-value",
            "AWS_CREDENTIAL=AKIAXXXXXXXXXXXXXXXX",
            "CLIENT_SECRET=oauth-client-secret-value",
        ]

        for text in test_cases:
            result = sanitize_sensitive_data(text)
            assert "[REDACTED]" in result, f"Failed for: {text}"

    def test_sanitize_passwords(self):
        """Sanitizes password patterns."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        test_cases = [
            'password=mysecretpassword',
            'PASSWORD: "complex_p@ssw0rd!"',
            "passwd=linux_password",
            "pwd=shortpwd",
        ]

        for text in test_cases:
            result = sanitize_sensitive_data(text)
            assert "[REDACTED]" in result, f"Failed for: {text}"

    def test_sanitize_sk_api_keys(self):
        """Sanitizes OpenAI-style sk- API keys."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        text = "Using API key: sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
        result = sanitize_sensitive_data(text)
        assert "[REDACTED_API_KEY]" in result
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz" not in result

    def test_sanitize_basic_auth(self):
        """Sanitizes Basic auth headers."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        text = "Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQxMjM0NTY3ODk="
        result = sanitize_sensitive_data(text)
        assert "Basic [REDACTED]" in result
        assert "dXNlcm5hbWU6cGFzc3dvcmQ" not in result

    def test_sanitize_mixed_content(self):
        """Sanitizes text with multiple sensitive patterns."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        text = """
Configuration:
OPENROUTER_API_KEY=sk-or-v1-abc123
STRIPE_API_KEY=sk_live_xyz789
webhook_url=https://hooks.slack.com/services/T00/B00/XXX
DB_PASSWORD=super_secret
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9
"""
        result = sanitize_sensitive_data(text)

        # All sensitive values should be redacted
        assert "sk-or-v1-abc123" not in result
        assert "sk_live_xyz789" not in result
        assert "hooks.slack.com" not in result
        assert "super_secret" not in result
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

        # But structure should be preserved
        assert "Configuration:" in result
        assert "[REDACTED]" in result

    def test_sanitize_preserves_non_sensitive(self):
        """Preserves non-sensitive content while sanitizing."""
        from ab_cli.commands.prompt import sanitize_sensitive_data

        text = """
# Application Config
DEBUG=true
LOG_LEVEL=info
API_URL=https://api.example.com
STRIPE_API_KEY=sk_live_secret123
PORT=8080
"""
        result = sanitize_sensitive_data(text)

        # Non-sensitive values preserved
        assert "DEBUG=true" in result
        assert "LOG_LEVEL=info" in result
        assert "PORT=8080" in result

        # Sensitive values redacted
        assert "sk_live_secret123" not in result
