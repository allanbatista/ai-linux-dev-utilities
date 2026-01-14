"""Unit tests for ab_cli.core.base_command module."""
import argparse
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from ab_cli.core import CliCommand
from ab_cli.core.base_command import CliCommand as CliCommandDirect


class ExampleCommand(CliCommand):
    """Example command implementation for testing."""

    def get_description(self) -> str:
        return "Example command for testing"

    def setup_arguments(self) -> None:
        self.parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Enable verbose output')
        self.parser.add_argument('--output', '-o', type=str,
                                 help='Output file path')
        self.parser.add_argument('input', nargs='?', help='Input file')

    def execute(self, args: argparse.Namespace) -> int:
        if args.verbose:
            print("Verbose mode enabled")
        if args.input:
            print(f"Processing: {args.input}")
        return 0


class FailingCommand(CliCommand):
    """Command that raises an exception for testing error handling."""

    def get_description(self) -> str:
        return "Command that fails"

    def setup_arguments(self) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        raise ValueError("Simulated error")


class ExitCodeCommand(CliCommand):
    """Command that returns specific exit codes."""

    def get_description(self) -> str:
        return "Command with exit codes"

    def setup_arguments(self) -> None:
        self.parser.add_argument('--code', type=int, default=0,
                                 help='Exit code to return')

    def execute(self, args: argparse.Namespace) -> int:
        return args.code


class TestCliCommandImport:
    """Tests for CliCommand import accessibility."""

    def test_import_from_core(self):
        """CliCommand can be imported from ab_cli.core."""
        from ab_cli.core import CliCommand as ImportedClass
        assert ImportedClass is CliCommandDirect

    def test_import_from_base_command(self):
        """CliCommand can be imported from ab_cli.core.base_command."""
        from ab_cli.core.base_command import CliCommand as DirectImport
        assert DirectImport is CliCommandDirect


class TestCliCommandAbstract:
    """Tests for CliCommand abstract base class requirements."""

    def test_cannot_instantiate_abstract_class(self):
        """CliCommand cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            CliCommand()
        assert "abstract" in str(exc_info.value).lower()

    def test_subclass_must_implement_get_description(self):
        """Subclass must implement get_description."""
        class IncompleteCommand(CliCommand):
            def setup_arguments(self): pass
            def execute(self, args): return 0

        with pytest.raises(TypeError):
            IncompleteCommand()

    def test_subclass_must_implement_setup_arguments(self):
        """Subclass must implement setup_arguments."""
        class IncompleteCommand(CliCommand):
            def get_description(self): return "test"
            def execute(self, args): return 0

        with pytest.raises(TypeError):
            IncompleteCommand()

    def test_subclass_must_implement_execute(self):
        """Subclass must implement execute."""
        class IncompleteCommand(CliCommand):
            def get_description(self): return "test"
            def setup_arguments(self): pass

        with pytest.raises(TypeError):
            IncompleteCommand()


class TestCliCommandInit:
    """Tests for CliCommand initialization."""

    def test_parser_created(self):
        """Command creates ArgumentParser on init."""
        cmd = ExampleCommand()
        assert isinstance(cmd.parser, argparse.ArgumentParser)

    def test_parser_has_description(self):
        """Parser has the correct description."""
        cmd = ExampleCommand()
        assert cmd.parser.description == "Example command for testing"

    def test_arguments_are_setup(self):
        """Arguments are configured during init."""
        cmd = ExampleCommand()
        # Parse to verify arguments exist
        args = cmd.parse_input(['--verbose', 'test.txt'])
        assert args.verbose is True
        assert args.input == 'test.txt'


class TestCliCommandParseInput:
    """Tests for CliCommand.parse_input() method."""

    def test_parse_input_with_args(self):
        """parse_input correctly parses provided arguments."""
        cmd = ExampleCommand()
        args = cmd.parse_input(['--verbose', '-o', 'out.txt', 'input.txt'])
        assert args.verbose is True
        assert args.output == 'out.txt'
        assert args.input == 'input.txt'

    def test_parse_input_empty_args(self):
        """parse_input handles empty argument list."""
        cmd = ExampleCommand()
        args = cmd.parse_input([])
        assert args.verbose is False
        assert args.output is None
        assert args.input is None

    def test_parse_input_defaults(self):
        """parse_input uses default values."""
        cmd = ExitCodeCommand()
        args = cmd.parse_input([])
        assert args.code == 0


class TestCliCommandExecute:
    """Tests for CliCommand.execute() method."""

    def test_execute_returns_exit_code(self):
        """execute returns the correct exit code."""
        cmd = ExitCodeCommand()
        args = cmd.parse_input(['--code', '42'])
        result = cmd.execute(args)
        assert result == 42

    def test_execute_success(self, capsys):
        """execute performs command logic."""
        cmd = ExampleCommand()
        args = cmd.parse_input(['--verbose', 'test.txt'])
        result = cmd.execute(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "Verbose mode enabled" in captured.out
        assert "Processing: test.txt" in captured.out


class TestCliCommandRun:
    """Tests for CliCommand.run() method."""

    def test_run_returns_success(self):
        """run returns 0 on success."""
        cmd = ExampleCommand()
        result = cmd.run(['input.txt'])
        assert result == 0

    def test_run_returns_custom_exit_code(self):
        """run returns custom exit code from execute."""
        cmd = ExitCodeCommand()
        result = cmd.run(['--code', '5'])
        assert result == 5

    def test_run_handles_exception(self, capsys):
        """run catches exceptions and returns 1."""
        cmd = FailingCommand()
        result = cmd.run([])

        captured = capsys.readouterr()
        assert result == 1
        assert "Error: Simulated error" in captured.err

    def test_run_handles_keyboard_interrupt(self, capsys):
        """run catches KeyboardInterrupt and returns 130."""
        class InterruptCommand(CliCommand):
            def get_description(self): return "test"
            def setup_arguments(self): pass
            def execute(self, args): raise KeyboardInterrupt()

        cmd = InterruptCommand()
        result = cmd.run([])

        captured = capsys.readouterr()
        assert result == 130
        assert "Operation cancelled" in captured.out

    def test_run_handles_system_exit(self):
        """run handles SystemExit from argparse (e.g., --help)."""
        cmd = ExampleCommand()
        # Invalid argument should cause argparse to exit
        result = cmd.run(['--invalid-flag'])
        assert result != 0  # Non-zero exit code

    def test_run_with_none_uses_empty_list(self):
        """run with None as args parses empty list."""
        cmd = ExampleCommand()
        # This should work without errors when args is None
        # (argparse will use sys.argv by default)
        with patch.object(sys, 'argv', ['test']):
            result = cmd.run(None)
            assert result == 0


class TestCliCommandHelp:
    """Tests for help text and parser configuration."""

    def test_help_includes_description(self, capsys):
        """--help includes command description."""
        cmd = ExampleCommand()
        result = cmd.run(['--help'])

        captured = capsys.readouterr()
        assert result == 0
        assert "Example command for testing" in captured.out

    def test_help_includes_arguments(self, capsys):
        """--help includes argument help text."""
        cmd = ExampleCommand()
        result = cmd.run(['--help'])

        captured = capsys.readouterr()
        assert "--verbose" in captured.out
        assert "--output" in captured.out
        assert "Enable verbose output" in captured.out


class TestCliCommandIntegration:
    """Integration tests demonstrating typical usage patterns."""

    def test_typical_main_entry_point(self, capsys):
        """Demonstrate typical main() entry point pattern."""
        def main():
            return ExampleCommand().run()

        # Simulate command line execution
        with patch.object(sys, 'argv', ['cmd', '--verbose', 'file.txt']):
            result = main()

        captured = capsys.readouterr()
        assert result == 0
        assert "Verbose mode enabled" in captured.out

    def test_testing_with_args(self):
        """Demonstrate testing with explicit args list."""
        cmd = ExitCodeCommand()

        # Test success case
        assert cmd.run(['--code', '0']) == 0

        # Test failure case
        assert cmd.run(['--code', '1']) == 1

    def test_error_output_goes_to_stderr(self, capsys):
        """Error messages are written to stderr."""
        cmd = FailingCommand()
        cmd.run([])

        captured = capsys.readouterr()
        assert captured.err.strip().startswith("Error:")
        assert "Simulated error" in captured.err
