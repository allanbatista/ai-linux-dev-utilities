"""
Base command class for standardized CLI command interfaces.

This module provides an abstract base class that all ab-cli commands can inherit
from to ensure consistent argument parsing, error handling, and execution flow.

Usage Example:
    ```python
    from ab_cli.core import CliCommand

    class MyCommand(CliCommand):
        def get_description(self) -> str:
            return "My custom command description"

        def setup_arguments(self) -> None:
            self.parser.add_argument('--verbose', '-v', action='store_true',
                                     help='Enable verbose output')
            self.parser.add_argument('input', nargs='?', help='Input file')

        def execute(self, args: argparse.Namespace) -> int:
            if args.verbose:
                print("Verbose mode enabled")
            # Command logic here
            return 0  # Success

    # Entry point
    def main():
        return MyCommand().run()
    ```

Benefits:
    - Consistent error handling across all commands
    - Standardized exit codes (0=success, 1=error, 130=cancelled)
    - Keyboard interrupt handling built-in
    - Easy testing with optional args parameter
    - Separation of concerns (parsing vs execution)
"""

import argparse
import sys
from abc import ABC, abstractmethod
from typing import Optional, List


class CliCommand(ABC):
    """
    Abstract base class for CLI commands.

    Provides a standardized interface for implementing CLI commands with
    consistent argument parsing, error handling, and execution flow.

    Subclasses must implement:
        - get_description(): Return the command description for --help
        - setup_arguments(): Configure argparse arguments
        - execute(args): Implement the command logic

    Attributes:
        parser (argparse.ArgumentParser): The argument parser instance

    Exit Codes:
        0: Success
        1: Error (exception or failure)
        130: Cancelled (KeyboardInterrupt)
    """

    def __init__(self):
        """Initialize the command with an argument parser."""
        self.parser = argparse.ArgumentParser(
            description=self.get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.setup_arguments()

    @abstractmethod
    def get_description(self) -> str:
        """
        Return the command description for help text.

        Returns:
            str: A description of what the command does.
        """
        pass

    @abstractmethod
    def setup_arguments(self) -> None:
        """
        Configure argparse arguments for this command.

        Example:
            ```python
            def setup_arguments(self) -> None:
                self.parser.add_argument('--output', '-o', help='Output file')
                self.parser.add_argument('input', help='Input file')
            ```
        """
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the command with parsed arguments.

        Args:
            args: Parsed command-line arguments from argparse

        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        pass

    def parse_input(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Args:
            args: Optional list of arguments. If None, uses sys.argv[1:]

        Returns:
            argparse.Namespace: Parsed arguments
        """
        return self.parser.parse_args(args)

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main entry point with standardized error handling.

        This method:
        1. Parses arguments (from args or sys.argv)
        2. Calls execute() with parsed args
        3. Handles exceptions and returns appropriate exit codes

        Args:
            args: Optional list of arguments for testing. If None, uses sys.argv[1:]

        Returns:
            int: Exit code (0=success, 1=error, 130=cancelled)

        Example:
            ```python
            # In main entry point
            def main():
                return MyCommand().run()

            # In tests
            def test_command():
                cmd = MyCommand()
                result = cmd.run(['--verbose', 'input.txt'])
                assert result == 0
            ```
        """
        try:
            parsed = self.parse_input(args)
            return self.execute(parsed)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return 130
        except SystemExit as e:
            # argparse calls sys.exit on --help or error
            return e.code if isinstance(e.code, int) else 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
