"""Integration tests for ab_cli.commands.auto_commit module."""
import subprocess
import sys
from unittest.mock import patch

import pytest

from ab_cli.commands.auto_commit import (
    create_commit,
    get_latest_commit,
    get_recent_commits,
    get_repo_root,
    get_staged_diff,
    get_staged_files,
    get_staged_name_status,
    get_unstaged_files,
    get_untracked_files,
    is_git_repo,
    main,
    stage_all_files,
)


class TestGitRepoDetection:
    """Tests for git repository detection."""

    def test_is_git_repo_true(self, mock_git_repo, monkeypatch):
        """Detects git repository correctly."""
        monkeypatch.chdir(mock_git_repo)
        assert is_git_repo() is True

    def test_is_git_repo_false(self, tmp_path, monkeypatch):
        """Returns False outside git repository."""
        monkeypatch.chdir(tmp_path)
        assert is_git_repo() is False


class TestStagedFiles:
    """Tests for staged file operations."""

    def test_get_staged_files_empty(self, mock_git_repo, monkeypatch):
        """Returns empty string when no files staged."""
        monkeypatch.chdir(mock_git_repo)
        result = get_staged_files()
        assert result == ""

    def test_get_staged_files_with_files(self, mock_git_repo, monkeypatch):
        """Returns staged file list."""
        monkeypatch.chdir(mock_git_repo)

        # Create and stage a file
        test_file = mock_git_repo / "test.txt"
        test_file.write_text("test content")
        subprocess.run(["git", "add", "test.txt"], cwd=mock_git_repo, check=True)

        result = get_staged_files()
        assert "test.txt" in result

    def test_get_staged_diff(self, mock_git_repo, monkeypatch):
        """Returns diff content for staged files."""
        monkeypatch.chdir(mock_git_repo)

        # Create and stage a file
        test_file = mock_git_repo / "test.txt"
        test_file.write_text("test content\n")
        subprocess.run(["git", "add", "test.txt"], cwd=mock_git_repo, check=True)

        result = get_staged_diff()
        assert "+test content" in result

    def test_get_staged_name_status(self, mock_git_repo, monkeypatch):
        """Returns staged files with status."""
        monkeypatch.chdir(mock_git_repo)

        # Create and stage a file
        test_file = mock_git_repo / "new_file.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "new_file.txt"], cwd=mock_git_repo, check=True)

        result = get_staged_name_status()
        assert "A" in result  # Added file
        assert "new_file.txt" in result


class TestUnstagedFiles:
    """Tests for unstaged file operations."""

    def test_get_unstaged_files_empty(self, mock_git_repo, monkeypatch):
        """Returns empty string when no unstaged changes."""
        monkeypatch.chdir(mock_git_repo)
        result = get_unstaged_files()
        assert result == ""

    def test_get_unstaged_files_with_changes(self, mock_git_repo, monkeypatch):
        """Returns modified files not staged."""
        monkeypatch.chdir(mock_git_repo)

        # Modify existing file
        readme = mock_git_repo / "README.md"
        readme.write_text("Modified content\n")

        result = get_unstaged_files()
        assert "README.md" in result


class TestUntrackedFiles:
    """Tests for untracked file operations."""

    def test_get_untracked_files_empty(self, mock_git_repo, monkeypatch):
        """Returns empty string when no untracked files."""
        monkeypatch.chdir(mock_git_repo)
        result = get_untracked_files()
        assert result == ""

    def test_get_untracked_files_with_files(self, mock_git_repo, monkeypatch):
        """Returns untracked file list."""
        monkeypatch.chdir(mock_git_repo)

        # Create untracked file
        new_file = mock_git_repo / "untracked.txt"
        new_file.write_text("new content")

        result = get_untracked_files()
        assert "untracked.txt" in result


class TestStageAndCommit:
    """Tests for staging and committing."""

    def test_stage_all_files(self, mock_git_repo, monkeypatch):
        """Stages all files with git add -A."""
        monkeypatch.chdir(mock_git_repo)

        # Create untracked files
        (mock_git_repo / "file1.txt").write_text("content1")
        (mock_git_repo / "file2.txt").write_text("content2")

        stage_all_files()

        # Check files are staged
        staged = get_staged_files()
        assert "file1.txt" in staged
        assert "file2.txt" in staged

    def test_create_commit(self, mock_git_repo, monkeypatch):
        """Creates git commit with message."""
        monkeypatch.chdir(mock_git_repo)

        # Create and stage a file
        (mock_git_repo / "test.txt").write_text("content")
        stage_all_files()

        create_commit("Test commit message")

        # Verify commit
        latest = get_latest_commit()
        assert "Test commit message" in latest

    def test_get_latest_commit(self, mock_git_repo, monkeypatch):
        """Returns latest commit in oneline format."""
        monkeypatch.chdir(mock_git_repo)
        result = get_latest_commit()
        assert "Initial commit" in result

    def test_get_recent_commits(self, mock_git_repo, monkeypatch):
        """Returns recent commit messages."""
        monkeypatch.chdir(mock_git_repo)

        # Create more commits
        for i in range(3):
            (mock_git_repo / f"file{i}.txt").write_text(f"content{i}")
            subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)
            subprocess.run(["git", "commit", "-m", f"Commit {i}"], cwd=mock_git_repo, check=True)

        result = get_recent_commits(5)
        assert "Commit 0" in result
        assert "Commit 1" in result
        assert "Commit 2" in result


class TestRepoRoot:
    """Tests for repository root detection."""

    def test_get_repo_root(self, mock_git_repo, monkeypatch):
        """Returns repository root directory."""
        monkeypatch.chdir(mock_git_repo)
        result = get_repo_root()
        assert result == str(mock_git_repo)

    def test_get_repo_root_from_subdir(self, mock_git_repo, monkeypatch):
        """Returns root from subdirectory."""
        subdir = mock_git_repo / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        result = get_repo_root()
        assert result == str(mock_git_repo)


class TestMain:
    """Tests for main() entry point."""

    def test_main_not_git_repo_exits_1(self, tmp_path, monkeypatch, capsys):
        """Exits with error when not in git repository."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["auto-commit"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Not inside a git repository" in captured.err

    def test_main_no_changes_exits_0(self, mock_git_repo, monkeypatch, capsys):
        """Exits cleanly when no changes to commit."""
        monkeypatch.chdir(mock_git_repo)
        monkeypatch.setattr(sys, "argv", ["auto-commit"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No changes to commit" in captured.out

    def test_main_prompt_not_found_exits_1(self, mock_git_repo, monkeypatch, capsys):
        """Exits with error if API call fails."""
        monkeypatch.chdir(mock_git_repo)

        # Create changes
        (mock_git_repo / "test.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-Y"])

        # Mock call_llm_with_model_info to return None (API failure)
        # Also mock is_protected_branch to avoid input() prompt
        with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
            with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=False):
                mock_call.return_value = (None, "test-model", 100)

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    @pytest.mark.parametrize("stage_flag", ["-y", "-a"])
    def test_main_auto_stage_flag(self, mock_git_repo, monkeypatch, stage_flag):
        """'-y' and '-a' flags stage all files."""
        monkeypatch.chdir(mock_git_repo)

        # Create unstaged file
        (mock_git_repo / "unstaged.txt").write_text("content")

        # Verify file is not staged initially
        assert "unstaged.txt" not in get_staged_files()

        monkeypatch.setattr(sys, "argv", ["auto-commit", stage_flag, "-Y"])

        # Mock call_llm_with_model_info to fail after staging happens
        call_count = [0]
        original_stage = stage_all_files

        def mock_stage():
            original_stage()
            call_count[0] += 1

        # Also mock is_protected_branch to avoid input() prompt
        with patch("ab_cli.commands.auto_commit.stage_all_files", side_effect=mock_stage):
            with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=False):
                    mock_call.return_value = (
                        '{"branch_name": "feature/test", "commit_message": "Test commit message"}',
                        "test-model",
                        100,
                    )

                    main()

        # Verify staging was called (the flag was honored)
        assert call_count[0] >= 1

    def test_main_user_cancels(self, mock_git_repo, monkeypatch, capsys):
        """Handles user cancellation during staging prompt."""
        monkeypatch.chdir(mock_git_repo)

        # Create unstaged file
        (mock_git_repo / "test.txt").write_text("content")

        monkeypatch.setattr(sys, "argv", ["auto-commit"])

        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_lang_flag(self, mock_git_repo, monkeypatch, capsys):
        """'-l' flag sets language."""
        monkeypatch.chdir(mock_git_repo)

        # Create and stage changes
        (mock_git_repo / "test.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-l", "pt-br", "-Y"])

        # Also mock is_protected_branch to avoid input() prompt
        with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
            with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=False):
                mock_call.return_value = (None, "test-model", 100)  # Fail to abort

                with pytest.raises(SystemExit):
                    main()

        captured = capsys.readouterr()
        # Language should be in info output
        assert "pt-br" in captured.out or captured.err

    def test_main_yes_commit_flag_skips_confirmation(self, mock_git_repo, monkeypatch):
        """'-Y' skips the final commit confirmation prompt."""
        monkeypatch.chdir(mock_git_repo)

        (mock_git_repo / "test.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-f", "-Y"])

        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=True):
                    mock_call.return_value = (
                        '{"branch_name": "feature/test", "commit_message": "Test commit message"}',
                        "test-model",
                        100,
                    )

                    main()

        assert "Test commit message" in get_latest_commit()

    def test_main_force_flag_skips_protected_branch_flow(self, mock_git_repo, monkeypatch):
        """'-f' keeps the current branch on protected branches."""
        monkeypatch.chdir(mock_git_repo)

        (mock_git_repo / "force.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-f", "-Y"])

        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            with patch("ab_cli.commands.auto_commit.handle_protected_branch") as mock_handle:
                with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                    with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=True):
                        mock_call.return_value = (
                            '{"branch_name": "feature/force-branch", "commit_message": "Force branch commit"}',
                            "test-model",
                            100,
                        )

                        main()

        mock_handle.assert_not_called()
        assert "Force branch commit" in get_latest_commit()

    def test_main_staged_only_ignores_unstaged_changes(self, mock_git_repo, monkeypatch):
        """'-s' uses only staged files and ignores unstaged/untracked changes."""
        monkeypatch.chdir(mock_git_repo)

        (mock_git_repo / "staged.txt").write_text("staged content\n")
        subprocess.run(["git", "add", "staged.txt"], cwd=mock_git_repo, check=True)

        (mock_git_repo / "unstaged.txt").write_text("unstaged content\n")
        (mock_git_repo / "untracked.txt").write_text("untracked content\n")

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-f", "-s", "-Y"])

        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            with patch("ab_cli.commands.auto_commit.stage_all_files") as mock_stage:
                with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                    with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=True):
                        mock_call.return_value = (
                            '{"branch_name": "feature/staged-only", "commit_message": "Staged only commit"}',
                            "test-model",
                            100,
                        )

                        main()

        mock_stage.assert_not_called()
        prompt_text = mock_call.call_args.args[0]
        assert "staged.txt" in prompt_text
        assert "unstaged.txt" not in prompt_text
        assert "untracked.txt" not in prompt_text
        assert "Staged only commit" in get_latest_commit()

    def test_main_push_flag_pushes_current_branch(self, mock_git_repo, monkeypatch):
        """'-p' pushes the current branch after committing."""
        monkeypatch.chdir(mock_git_repo)

        subprocess.run(["git", "checkout", "-b", "feature/push"], cwd=mock_git_repo, check=True)
        (mock_git_repo / "push.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-p", "-Y"])

        with patch("ab_cli.commands.auto_commit.push_branch", return_value=True) as mock_push:
            with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=False):
                    mock_call.return_value = (
                        '{"branch_name": "feature/push", "commit_message": "Push commit"}',
                        "test-model",
                        100,
                    )

                    main()

        mock_push.assert_called_once_with("feature/push")
        assert "Push commit" in get_latest_commit()

    def test_main_pr_flag_creates_pr_after_push(self, mock_git_repo, monkeypatch):
        """'-P' creates a PR after pushing the branch."""
        monkeypatch.chdir(mock_git_repo)

        subprocess.run(["git", "checkout", "-b", "feature/pr"], cwd=mock_git_repo, check=True)
        (mock_git_repo / "pr.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-p", "-P", "-Y"])

        with patch("ab_cli.commands.auto_commit.push_branch", return_value=True) as mock_push:
            with patch("ab_cli.commands.auto_commit.check_gh_installed", return_value=True):
                with patch("ab_cli.commands.auto_commit.check_gh_authenticated", return_value=True):
                    with patch("ab_cli.commands.auto_commit.generate_pr_content", return_value=("PR title", "PR body")) as mock_pr_content:
                        with patch("ab_cli.commands.auto_commit.create_pr", return_value="https://example.com/pr/1") as mock_create_pr:
                            with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                                with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=False):
                                    mock_call.return_value = (
                                        '{"branch_name": "feature/pr", "commit_message": "PR commit"}',
                                        "test-model",
                                        100,
                                    )

                                    main()

        mock_push.assert_called_once_with("feature/pr")
        mock_pr_content.assert_called_once()
        mock_create_pr.assert_called_once_with("PR title", "PR body", "master")
        assert "PR commit" in get_latest_commit()

    def test_main_pr_flag_creates_branch_from_protected_master(self, mock_git_repo, monkeypatch):
        """'-y -Y -p -P' creates the suggested branch from master without prompting."""
        monkeypatch.chdir(mock_git_repo)
        subprocess.run(["git", "checkout", "-B", "master"], cwd=mock_git_repo, check=True)
        (mock_git_repo / "protected-master.txt").write_text("content")

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-y", "-Y", "-p", "-P"])

        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            with patch("ab_cli.commands.auto_commit.push_branch", return_value=True) as mock_push:
                with patch("ab_cli.commands.auto_commit.check_gh_installed", return_value=True):
                    with patch("ab_cli.commands.auto_commit.check_gh_authenticated", return_value=True):
                        with patch("ab_cli.commands.auto_commit.detect_base_branch", return_value="master"):
                            with patch("ab_cli.commands.auto_commit.generate_pr_content", return_value=("PR title", "PR body")):
                                with patch("ab_cli.commands.auto_commit.create_pr", return_value="https://example.com/pr/1") as mock_create_pr:
                                    with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                                        mock_call.return_value = (
                                            '{"branch_name": "feature/protected-master", "commit_message": "Master PR commit"}',
                                            "test-model",
                                            100,
                                        )

                                        main()

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=mock_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert branch == "feature/protected-master"
        mock_push.assert_called_once_with("feature/protected-master")
        mock_create_pr.assert_called_once_with("PR title", "PR body", "master")
        assert "Master PR commit" in get_latest_commit()

    def test_main_pr_flag_creates_branch_from_protected_main(self, mock_git_repo, monkeypatch):
        """'-y -Y -p -P' creates the suggested branch from main without prompting."""
        monkeypatch.chdir(mock_git_repo)
        subprocess.run(["git", "checkout", "-B", "main"], cwd=mock_git_repo, check=True)
        (mock_git_repo / "protected-main.txt").write_text("content")

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-y", "-Y", "-p", "-P"])

        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            with patch("ab_cli.commands.auto_commit.push_branch", return_value=True) as mock_push:
                with patch("ab_cli.commands.auto_commit.check_gh_installed", return_value=True):
                    with patch("ab_cli.commands.auto_commit.check_gh_authenticated", return_value=True):
                        with patch("ab_cli.commands.auto_commit.detect_base_branch", return_value="main"):
                            with patch("ab_cli.commands.auto_commit.generate_pr_content", return_value=("PR title", "PR body")):
                                with patch("ab_cli.commands.auto_commit.create_pr", return_value="https://example.com/pr/1") as mock_create_pr:
                                    with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                                        mock_call.return_value = (
                                            '{"branch_name": "feature/protected-main", "commit_message": "Main PR commit"}',
                                            "test-model",
                                            100,
                                        )

                                        main()

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=mock_git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert branch == "feature/protected-main"
        mock_push.assert_called_once_with("feature/protected-main")
        mock_create_pr.assert_called_once_with("PR title", "PR body", "main")
        assert "Main PR commit" in get_latest_commit()

    def test_main_pr_force_on_protected_branch_fails_before_push(self, mock_git_repo, monkeypatch, capsys):
        """'-f -y -Y -p -P' does not auto-branch and cannot create PR from master."""
        monkeypatch.chdir(mock_git_repo)
        subprocess.run(["git", "checkout", "-B", "master"], cwd=mock_git_repo, check=True)
        (mock_git_repo / "force-pr.txt").write_text("content")

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-f", "-y", "-Y", "-p", "-P"])

        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            with patch("ab_cli.commands.auto_commit.create_branch") as mock_create_branch:
                with patch("ab_cli.commands.auto_commit.push_branch") as mock_push:
                    with patch("ab_cli.commands.auto_commit.create_pr") as mock_create_pr:
                        with patch("ab_cli.commands.auto_commit.call_llm_with_model_info") as mock_call:
                            mock_call.return_value = (
                                '{"branch_name": "feature/force-pr", "commit_message": "Force PR commit"}',
                                "test-model",
                                100,
                            )

                            with pytest.raises(SystemExit) as exc_info:
                                main()

        assert exc_info.value.code == 1
        assert "-P requires a non-protected branch" in capsys.readouterr().err
        mock_create_branch.assert_not_called()
        mock_push.assert_not_called()
        mock_create_pr.assert_not_called()

    def test_main_pr_flag_with_clean_tree_uses_pr_flow(self, mock_git_repo, monkeypatch):
        """'-P' runs PR flow even when there are no working tree edits."""
        monkeypatch.chdir(mock_git_repo)

        subprocess.run(["git", "checkout", "-b", "feature/pr-clean"], cwd=mock_git_repo, check=True)
        (mock_git_repo / "pr-clean.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=mock_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Clean tree commit"], cwd=mock_git_repo, check=True)

        monkeypatch.setattr(sys, "argv", ["auto-commit", "-p", "-P", "-Y"])

        with patch("ab_cli.commands.auto_commit.push_branch", return_value=True) as mock_push:
            with patch("ab_cli.commands.auto_commit.check_gh_installed", return_value=True):
                with patch("ab_cli.commands.auto_commit.check_gh_authenticated", return_value=True):
                    with patch("ab_cli.commands.auto_commit.generate_pr_content", return_value=("PR title", "PR body")) as mock_pr_content:
                        with patch("ab_cli.commands.auto_commit.create_pr", return_value="https://example.com/pr/1") as mock_create_pr:
                            with patch("ab_cli.commands.auto_commit.call_llm_with_model_info", side_effect=AssertionError("commit plan should not be generated")):
                                with patch("ab_cli.commands.auto_commit.is_protected_branch", return_value=False):
                                    main()

        mock_push.assert_called_once_with("feature/pr-clean")
        mock_pr_content.assert_called_once()
        mock_create_pr.assert_called_once_with("PR title", "PR body", "master")
        assert "Clean tree commit" in get_latest_commit()

    def test_main_pr_flag_without_push_exits_1(self, mock_git_repo, monkeypatch, capsys):
        """'-P' requires '-p'."""
        monkeypatch.chdir(mock_git_repo)
        monkeypatch.setattr(sys, "argv", ["auto-commit", "-P"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "-P requires -p" in captured.err
