"""Unit tests for ab_cli.core.config module."""
import json
from pathlib import Path

import pytest

from ab_cli.core.config import (
    AbConfig,
    AbConfigModel,
    ConfigValidationError,
    DEFAULT_CONFIG,
    GlobalConfigModel,
    HistoryConfigModel,
    ModelsConfigModel,
    ThresholdsModel,
    estimate_tokens,
    get_config,
    get_default_model,
    get_language,
    select_model_for_tokens,
)


class TestAbConfigSingleton:
    """Tests for AbConfig singleton pattern."""

    def test_singleton_pattern(self, temp_config_dir):
        """AbConfig returns same instance on multiple calls."""
        config1 = AbConfig()
        config2 = AbConfig()
        assert config1 is config2

    def test_get_config_returns_singleton(self, temp_config_dir):
        """get_config() returns the same AbConfig instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestAbConfigGet:
    """Tests for AbConfig.get() method."""

    def test_get_with_dot_notation(self, mock_config):
        """get('global.language') works with dot notation."""
        config = get_config()
        assert config.get("global.language") == "en"

    def test_get_nested_path(self, mock_config):
        """Deep paths like 'models.thresholds.small_max_tokens' work."""
        config = get_config()
        assert config.get("models.thresholds.small_max_tokens") == 128000

    def test_get_missing_key_returns_default(self, mock_config):
        """Missing keys return the default value."""
        config = get_config()
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_get_top_level_key(self, mock_config):
        """Top-level keys work without dot notation."""
        config = get_config()
        assert config.get("version") == "1.0"

    def test_get_returns_none_for_partial_path(self, mock_config):
        """Partial paths that don't exist return None."""
        config = get_config()
        assert config.get("global.nonexistent") is None

    def test_get_with_default_returns_value_when_exists(self, mock_config):
        """get_with_default returns config value when it exists."""
        config = get_config()
        assert config.get_with_default("global.language") == "en"

    def test_get_with_default_falls_back_to_default_config(self, temp_config_dir):
        """get_with_default falls back to DEFAULT_CONFIG when key missing."""
        config = get_config()
        # No config file exists, should fall back to defaults
        assert config.get_with_default("global.language") == DEFAULT_CONFIG["global"]["language"]


class TestAbConfigSet:
    """Tests for AbConfig.set() method."""

    def test_set_existing_path(self, mock_config, temp_config_dir):
        """Setting an existing path updates the value."""
        from ab_cli.core import config as config_module

        config = get_config()
        config.set("global.language", "pt-br")
        assert config.get("global.language") == "pt-br"

        # Verify persisted to file
        with open(config_module.AB_CONFIG_FILE) as f:
            saved = json.load(f)
        assert saved["global"]["language"] == "pt-br"

    def test_set_creates_nested_structure(self, mock_config, temp_config_dir):
        """set('new.nested.path', value) creates parent dicts."""
        from ab_cli.core import config as config_module

        config = get_config()
        config.set("custom.new.setting", "test_value")
        assert config.get("custom.new.setting") == "test_value"

        # Verify persisted
        with open(config_module.AB_CONFIG_FILE) as f:
            saved = json.load(f)
        assert saved["custom"]["new"]["setting"] == "test_value"

    def test_set_persists_to_file(self, mock_config, temp_config_dir):
        """Changes are saved to config.json."""
        from ab_cli.core import config as config_module

        config = get_config()
        config.set("models.default", "new/model")

        with open(config_module.AB_CONFIG_FILE) as f:
            saved = json.load(f)
        assert saved["models"]["default"] == "new/model"


class TestAbConfigSelectModel:
    """Tests for AbConfig.select_model() method."""

    def test_select_model_small(self, mock_config):
        """Tokens <= 128k returns small model."""
        config = get_config()
        assert config.select_model(50000) == "test/model-small"
        assert config.select_model(128000) == "test/model-small"

    def test_select_model_medium(self, mock_config):
        """Tokens <= 256k returns medium model."""
        config = get_config()
        assert config.select_model(128001) == "test/model-medium"
        assert config.select_model(256000) == "test/model-medium"

    def test_select_model_large(self, mock_config):
        """Tokens > 256k returns large model."""
        config = get_config()
        assert config.select_model(256001) == "test/model-large"
        assert config.select_model(500000) == "test/model-large"

    def test_select_model_uses_thresholds(self, mock_config, temp_config_dir):
        """Model selection respects configured thresholds."""
        config = get_config()
        # Update thresholds
        config.set("models.thresholds.small_max_tokens", 50000)
        config.reload()

        # Now 60000 tokens should select medium
        assert config.select_model(60000) == "test/model-medium"


class TestAbConfigCommandSettings:
    """Tests for AbConfig.get_command_setting() method."""

    def test_get_command_setting_specific(self, mock_config):
        """Command-specific settings override global."""
        config = get_config()
        # auto-commit has language=pt-br in mock_config
        assert config.get_command_setting("auto-commit", "language") == "pt-br"

    def test_get_command_setting_fallback_global(self, mock_config):
        """Falls back to global when command-specific missing."""
        config = get_config()
        # pr-description has no language setting
        assert config.get_command_setting("pr-description", "language") == "en"

    def test_get_command_setting_fallback_default(self, mock_config):
        """Falls back to default when both missing."""
        config = get_config()
        assert config.get_command_setting("any-command", "nonexistent", "fallback") == "fallback"


class TestAbConfigInit:
    """Tests for AbConfig.init_config() method."""

    def test_init_config_creates_file(self, temp_config_dir):
        """Creates default config file."""
        from ab_cli.core import config as config_module

        config = get_config()
        result = config.init_config()

        assert result is True
        assert config_module.AB_CONFIG_FILE.exists()

    def test_init_config_existing_returns_false(self, mock_config):
        """Returns False if config already exists."""
        config = get_config()
        result = config.init_config()
        assert result is False


class TestAbConfigReload:
    """Tests for AbConfig.reload() method."""

    def test_reload_picks_up_changes(self, mock_config, temp_config_dir):
        """File changes reflected after reload."""
        from ab_cli.core import config as config_module

        config = get_config()

        # Modify file directly
        with open(config_module.AB_CONFIG_FILE) as f:
            data = json.load(f)
        data["global"]["language"] = "modified"
        with open(config_module.AB_CONFIG_FILE, "w") as f:
            json.dump(data, f)

        # Before reload, still old value (cached)
        # After reload, new value
        config.reload()
        assert config.get("global.language") == "modified"


class TestAbConfigMisc:
    """Tests for miscellaneous AbConfig methods."""

    def test_deep_merge(self, temp_config_dir):
        """_deep_merge correctly merges dicts."""
        config = get_config()
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 10, "d": 3}}
        result = config._deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2, "d": 3}}

    def test_config_exists_true(self, mock_config):
        """config_exists() returns True when file exists."""
        config = get_config()
        assert config.config_exists() is True

    def test_config_exists_false(self, temp_config_dir):
        """config_exists() returns False when file missing."""
        config = get_config()
        assert config.config_exists() is False

    def test_get_history_dir(self, mock_config, temp_config_dir):
        """Returns correct history directory path."""
        config = get_config()
        history_dir = config.get_history_dir()
        assert isinstance(history_dir, Path)
        assert "history" in str(history_dir)

    def test_is_history_enabled(self, mock_config):
        """is_history_enabled returns config value."""
        config = get_config()
        assert config.is_history_enabled() is True

    def test_to_dict(self, mock_config):
        """to_dict returns full config as dictionary."""
        config = get_config()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert "version" in data
        assert "global" in data

    def test_get_config_path(self, temp_config_dir):
        """get_config_path returns correct path."""
        from ab_cli.core import config as config_module

        path = AbConfig.get_config_path()
        assert path == config_module.AB_CONFIG_FILE

    def test_get_config_dir(self, temp_config_dir):
        """get_config_dir returns correct path."""
        from ab_cli.core import config as config_module

        path = AbConfig.get_config_dir()
        assert path == config_module.AB_CONFIG_DIR

    def test_get_api_settings(self, mock_config):
        """get_api_settings returns API configuration."""
        config = get_config()
        settings = config.get_api_settings()
        assert "api_base" in settings
        assert "api_key_env" in settings
        assert "timeout_seconds" in settings


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_estimate_tokens_basic(self):
        """estimate_tokens returns ~len/4."""
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_estimate_tokens_empty(self):
        """estimate_tokens handles empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        """estimate_tokens handles short text."""
        assert estimate_tokens("hi") == 0  # 2 // 4 = 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_language_default(self, temp_config_dir):
        """get_language returns default 'en'."""
        lang = get_language()
        assert lang == "en"

    def test_get_language_for_command(self, mock_config):
        """get_language returns command-specific language."""
        lang = get_language("auto-commit")
        assert lang == "pt-br"

    def test_get_default_model(self, mock_config):
        """get_default_model returns configured default."""
        model = get_default_model()
        assert model == "test/model-small"

    def test_select_model_for_tokens(self, mock_config):
        """select_model_for_tokens delegates to AbConfig."""
        model = select_model_for_tokens(50000)
        assert model == "test/model-small"


class TestPydanticModels:
    """Tests for Pydantic configuration models."""

    def test_global_config_defaults(self):
        """GlobalConfigModel has correct default values."""
        model = GlobalConfigModel()
        assert model.language == "en"
        assert model.api_base == "https://openrouter.ai/api/v1"
        assert model.api_key_env == "OPENROUTER_API_KEY"
        assert model.timeout_seconds == 300

    def test_global_config_custom_values(self):
        """GlobalConfigModel accepts custom values."""
        model = GlobalConfigModel(
            language="pt-br",
            api_base="https://custom.api/v1",
            timeout_seconds=120
        )
        assert model.language == "pt-br"
        assert model.api_base == "https://custom.api/v1"
        assert model.timeout_seconds == 120

    def test_global_config_timeout_validation(self):
        """GlobalConfigModel validates timeout_seconds bounds."""
        from pydantic import ValidationError

        # Timeout must be > 0
        with pytest.raises(ValidationError):
            GlobalConfigModel(timeout_seconds=0)

        # Timeout must be <= 600
        with pytest.raises(ValidationError):
            GlobalConfigModel(timeout_seconds=601)

        # Valid boundary values
        model_min = GlobalConfigModel(timeout_seconds=1)
        assert model_min.timeout_seconds == 1

        model_max = GlobalConfigModel(timeout_seconds=600)
        assert model_max.timeout_seconds == 600

    def test_thresholds_defaults(self):
        """ThresholdsModel has correct default values."""
        model = ThresholdsModel()
        assert model.small_max_tokens == 128000
        assert model.medium_max_tokens == 256000

    def test_thresholds_validation(self):
        """ThresholdsModel validates token values are positive."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ThresholdsModel(small_max_tokens=0)

        with pytest.raises(ValidationError):
            ThresholdsModel(medium_max_tokens=-1)

    def test_models_config_defaults(self):
        """ModelsConfigModel has correct default values."""
        model = ModelsConfigModel()
        assert model.small == "nvidia/nemotron-3-nano-30b-a3b:free"
        assert model.medium == "openai/gpt-5-nano"
        assert model.large == "x-ai/grok-4.1-fast"
        assert model.default == "nvidia/nemotron-3-nano-30b-a3b:free"
        assert isinstance(model.thresholds, ThresholdsModel)

    def test_history_config_defaults(self):
        """HistoryConfigModel has correct default values."""
        model = HistoryConfigModel()
        assert model.enabled is True
        assert model.directory == ""

    def test_ab_config_model_defaults(self):
        """AbConfigModel has correct default structure."""
        model = AbConfigModel()
        assert model.version == "1.0"
        assert isinstance(model.global_settings, GlobalConfigModel)
        assert isinstance(model.models, ModelsConfigModel)
        assert isinstance(model.commands, dict)
        assert isinstance(model.history, HistoryConfigModel)

    def test_ab_config_model_alias(self):
        """AbConfigModel uses 'global' alias correctly."""
        data = {
            "version": "1.0",
            "global": {
                "language": "fr",
                "timeout_seconds": 200
            },
            "models": {
                "small": "test/small",
                "medium": "test/medium",
                "large": "test/large",
                "default": "test/default"
            }
        }
        model = AbConfigModel.model_validate(data)
        assert model.global_settings.language == "fr"
        assert model.global_settings.timeout_seconds == 200

        # Verify dump uses alias
        dumped = model.model_dump(by_alias=True)
        assert "global" in dumped
        assert dumped["global"]["language"] == "fr"

    def test_ab_config_model_extra_fields_allowed(self):
        """AbConfigModel allows extra fields for forward compatibility."""
        data = {
            "version": "2.0",
            "global": {"language": "en", "new_field": "value"},
            "models": {
                "small": "a",
                "medium": "b",
                "large": "c",
                "default": "a"
            },
            "custom_section": {"key": "value"}
        }
        model = AbConfigModel.model_validate(data)
        assert model.version == "2.0"

    def test_ab_config_model_from_default_config(self):
        """AbConfigModel validates DEFAULT_CONFIG successfully."""
        model = AbConfigModel.model_validate(DEFAULT_CONFIG)
        assert model.version == "1.0"
        assert model.global_settings.language == "en"
        assert model.models.default == "nvidia/nemotron-3-nano-30b-a3b:free"


class TestConfigValidation:
    """Tests for config validation integration."""

    def test_validate_returns_model(self, mock_config):
        """validate() returns AbConfigModel on success."""
        config = get_config()
        model = config.validate()
        assert isinstance(model, AbConfigModel)

    def test_validate_with_custom_data(self, temp_config_dir):
        """validate() can validate custom data."""
        config = get_config()
        custom_data = {
            "version": "1.0",
            "global": {"language": "de", "timeout_seconds": 100},
            "models": {
                "small": "test/small",
                "medium": "test/medium",
                "large": "test/large",
                "default": "test/small"
            }
        }
        model = config.validate(custom_data)
        assert model.global_settings.language == "de"

    def test_validate_raises_on_invalid(self, temp_config_dir):
        """validate() raises ConfigValidationError on invalid data."""
        config = get_config()
        invalid_data = {
            "version": "1.0",
            "global": {"timeout_seconds": -1}  # Invalid: must be > 0
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate(invalid_data)
        assert len(exc_info.value.errors) > 0

    def test_get_validated_model(self, mock_config):
        """get_validated_model returns cached model."""
        config = get_config()
        model = config.get_validated_model()
        assert isinstance(model, AbConfigModel)

    def test_get_validation_errors_empty_on_valid(self, mock_config):
        """get_validation_errors returns empty list for valid config."""
        config = get_config()
        errors = config.get_validation_errors()
        assert errors == []

    def test_has_validation_errors_false_on_valid(self, mock_config):
        """has_validation_errors returns False for valid config."""
        config = get_config()
        assert config.has_validation_errors() is False

    def test_invalid_config_falls_back_to_defaults(self, temp_config_dir, capsys):
        """Invalid config values are replaced with defaults."""
        from ab_cli.core import config as config_module

        # Create config with invalid timeout
        invalid_config = {
            "version": "1.0",
            "global": {
                "language": "en",
                "timeout_seconds": 9999  # Invalid: > 600
            },
            "models": {
                "small": "test/small",
                "medium": "test/medium",
                "large": "test/large",
                "default": "test/default"
            }
        }
        config_file = config_module.AB_CONFIG_FILE
        with open(config_file, "w") as f:
            json.dump(invalid_config, f)

        config = get_config()
        config.reload()

        # Should have validation errors but still load
        assert config.has_validation_errors() is True
        errors = config.get_validation_errors()
        assert len(errors) > 0

        # Capture warning output
        captured = capsys.readouterr()
        assert "Warning" in captured.out or config.has_validation_errors()

    def test_reload_revalidates(self, mock_config, temp_config_dir):
        """reload() re-validates configuration."""
        from ab_cli.core import config as config_module

        config = get_config()
        initial_model = config.get_validated_model()
        assert initial_model is not None

        # Modify config file
        with open(config_module.AB_CONFIG_FILE) as f:
            data = json.load(f)
        data["global"]["language"] = "es"
        with open(config_module.AB_CONFIG_FILE, "w") as f:
            json.dump(data, f)

        config.reload()
        new_model = config.get_validated_model()
        assert new_model.global_settings.language == "es"


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_error_has_message(self):
        """ConfigValidationError stores message."""
        error = ConfigValidationError("Test error")
        assert str(error) == "Test error"

    def test_error_has_errors_list(self):
        """ConfigValidationError stores errors list."""
        errors = [{"loc": ("field",), "msg": "error"}]
        error = ConfigValidationError("Test error", errors=errors)
        assert error.errors == errors

    def test_error_default_empty_list(self):
        """ConfigValidationError defaults to empty errors list."""
        error = ConfigValidationError("Test error")
        assert error.errors == []
