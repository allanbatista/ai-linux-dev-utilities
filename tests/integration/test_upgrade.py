"""Integration tests for ab upgrade."""
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AB = ROOT / "bin" / "ab"
UPGRADE = ROOT / "bin" / "ab-upgrade"
COMPLETION = ROOT / "completions" / "ab.bash-completion"


def run_cmd(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def make_install(tmp_path):
    project = tmp_path / "ab-install"
    (project / "bin").mkdir(parents=True)
    (project / "completions").mkdir()
    shutil.copy2(UPGRADE, project / "bin" / "ab-upgrade")
    (project / "bin" / "ab-upgrade").chmod(0o755)
    (project / "bin" / "ab").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project / "bin" / "ab").chmod(0o755)
    (project / "requirements.txt").write_text("", encoding="utf-8")
    (project / "completions" / "ab.bash-completion").write_text("# completion\n", encoding="utf-8")
    return project


def make_stubs(tmp_path):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()

    (fakebin / "git").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "git $*" >> "$AB_UPGRADE_LOG"
if [ "$1" = "-C" ]; then
    shift 2
fi
case "$1" in
    rev-parse)
        echo true
        ;;
    status)
        if [ -n "${AB_DIRTY_STATUS:-}" ]; then
            echo "$AB_DIRTY_STATUS"
        fi
        ;;
    fetch)
        if [ "${AB_FAIL_STEP:-}" = "git-fetch" ]; then
            exit 42
        fi
        ;;
    pull)
        ;;
esac
""",
        encoding="utf-8",
    )
    (fakebin / "git").chmod(0o755)

    (fakebin / "python3").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "python3 $*" >> "$AB_UPGRADE_LOG"
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    mkdir -p "$3/bin"
    cat > "$3/bin/python" <<'PY'
#!/usr/bin/env bash
echo "venv-python $*" >> "$AB_UPGRADE_LOG"
if [ "${AB_FAIL_STEP:-}" = "pip-install" ] && [[ "$*" == *" install -r "* ]]; then
    exit 43
fi
PY
    chmod +x "$3/bin/python"
fi
""",
        encoding="utf-8",
    )
    (fakebin / "python3").chmod(0o755)

    return fakebin


def env_for(tmp_path, project):
    log = tmp_path / "upgrade.log"
    env = os.environ.copy()
    env["PATH"] = f"{make_stubs(tmp_path)}:{env['PATH']}"
    env["AB_UPGRADE_LOG"] = str(log)
    env["AB_INSTALL_PATH"] = str(tmp_path / "links" / "ab")
    env["AB_COMPLETION_FILE"] = str(tmp_path / "completions" / "ab")
    return env, log


def test_ab_help_lists_upgrade():
    result = run_cmd([str(AB), "help"])

    assert result.returncode == 0
    assert "upgrade" in result.stdout
    assert "Update this ab installation" in result.stdout


def test_upgrade_help_documents_behavior():
    result = run_cmd([str(AB), "upgrade", "--help"])

    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "non-interactively" in result.stdout
    assert "Preserves local changes" in result.stdout
    assert "Returns non-zero" in result.stdout


def test_completion_suggests_upgrade_and_options():
    level1_script = (
        f"source {COMPLETION}; "
        "COMP_WORDS=(ab up); COMP_CWORD=1; "
        "_ab_completions; printf '%s\\n' \"${COMPREPLY[@]}\""
    )
    options_script = (
        f"source {COMPLETION}; "
        "COMP_WORDS=(ab upgrade --); COMP_CWORD=2; "
        "_ab_completions; printf '%s\\n' \"${COMPREPLY[@]}\""
    )
    level1 = run_cmd([
        "bash",
        "-lc",
        level1_script,
    ])
    options = run_cmd([
        "bash",
        "-lc",
        options_script,
    ])

    assert level1.returncode == 0
    assert "upgrade" in level1.stdout
    assert options.returncode == 0
    assert "--dry-run" in options.stdout


def test_upgrade_runs_noninteractive_flow(tmp_path):
    project = make_install(tmp_path)
    env, log = env_for(tmp_path, project)

    result = run_cmd([str(project / "bin" / "ab-upgrade")], env=env)

    assert result.returncode == 0, result.stderr
    entries = log.read_text(encoding="utf-8").splitlines()
    assert entries == [
        f"git -C {project} rev-parse --is-inside-work-tree",
        f"git -C {project} status --porcelain --untracked-files=all",
        f"git -C {project} fetch --prune origin",
        f"git -C {project} pull --ff-only",
        f"python3 -m venv {project}/.venv",
        "venv-python -m pip install --upgrade pip -q",
        f"venv-python -m pip install -r {project}/requirements.txt -q",
    ]
    assert Path(env["AB_INSTALL_PATH"]).resolve() == project / "bin" / "ab"
    assert Path(env["AB_COMPLETION_FILE"]).resolve() == project / "completions" / "ab.bash-completion"
    assert "Upgrade complete" in result.stdout


def test_upgrade_dry_run_does_not_change_state(tmp_path):
    project = make_install(tmp_path)
    subprocess.run(["git", "init"], cwd=project, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project, check=True)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=project, check=True, capture_output=True)
    before_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=project).stdout
    before_status = run_cmd(["git", "status", "--porcelain", "--untracked-files=all"], cwd=project).stdout

    env = os.environ.copy()
    env["AB_INSTALL_PATH"] = str(tmp_path / "links" / "ab")
    env["AB_COMPLETION_FILE"] = str(tmp_path / "completions" / "ab")
    result = run_cmd([str(project / "bin" / "ab-upgrade"), "--dry-run"], env=env)

    assert result.returncode == 0, result.stderr
    assert before_head == run_cmd(["git", "rev-parse", "HEAD"], cwd=project).stdout
    assert before_status == run_cmd(["git", "status", "--porcelain", "--untracked-files=all"], cwd=project).stdout
    assert not (project / ".venv").exists()
    assert not Path(env["AB_INSTALL_PATH"]).exists()
    assert not Path(env["AB_COMPLETION_FILE"]).exists()
    assert "Dry run: no changes will be made" in result.stdout
    assert "git -C" in result.stdout
    assert "fetch --prune origin" in result.stdout


def test_upgrade_stops_on_failed_step(tmp_path):
    project = make_install(tmp_path)
    env, log = env_for(tmp_path, project)
    env["AB_FAIL_STEP"] = "git-fetch"

    result = run_cmd([str(project / "bin" / "ab-upgrade")], env=env)

    assert result.returncode != 0
    assert "failed during git fetch" in result.stderr
    entries = log.read_text(encoding="utf-8").splitlines()
    assert f"git -C {project} pull --ff-only" not in entries
    assert "venv-python -m pip install --upgrade pip -q" not in entries


def test_upgrade_preserves_dirty_worktree(tmp_path):
    project = make_install(tmp_path)
    env, log = env_for(tmp_path, project)
    env["AB_DIRTY_STATUS"] = " M README.md"

    result = run_cmd([str(project / "bin" / "ab-upgrade")], env=env)

    assert result.returncode != 0
    assert "worktree is not clean" in result.stderr
    assert " M README.md" in result.stderr
    entries = log.read_text(encoding="utf-8").splitlines()
    assert entries == [
        f"git -C {project} rev-parse --is-inside-work-tree",
        f"git -C {project} status --porcelain --untracked-files=all",
    ]
    assert not (project / ".venv").exists()
    assert not Path(env["AB_INSTALL_PATH"]).exists()
