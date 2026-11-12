"""Testes de integração dos comandos personalizados migrados."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


@pytest.mark.parametrize(
    ("command", "args", "status", "expected"),
    [
        ("avi-to-mp4", ["--help"], 0, "Converte AVI"),
        ("gh-loc-chart", ["--help"], 1, "Uso: gh-loc-chart"),
        ("git-cleanup", ["--help"], 2, "Opção desconhecida"),
        ("git-prune", ["--help"], 0, "Uso: git-prune"),
        ("graphify-project-setup", ["--help"], 0, "Usage: graphify-project-setup"),
        ("graphify-project-sync", ["--help"], 0, "Usage: graphify-project-sync"),
        ("graphify-update-and-commit", ["--help"], 0, "Usage: graphify-update-and-commit"),
        ("httpserver", ["--help"], 0, "Serve arquivos HTML"),
        ("image-normalizer", ["--help"], 0, "Normaliza imagem"),
        ("img-remove-croma", ["--help"], 0, "Normaliza imagem"),
        ("kill-service-by-ports", [], 1, "Uso: kill-service-by-ports"),
        ("mkworktree", [], 1, "Uso: mkworktree"),
        ("open-design", [], 1, "Open Design CLI is not built"),
        ("open-design-mcp", [], 1, "Open Design CLI is not built"),
        ("rm-node-modules", ["--help"], 0, "Uso: rm-node-modules"),
        ("rmworktree", ["--help"], 2, "Uso: rmworktree"),
        ("video_to_whatsapp", [], 1, "Uso: video_to_whatsapp"),
    ],
)
def test_commands_launch_through_symlinks(tmp_path, command, args, status, expected):
    links = tmp_path / "command links"
    links.mkdir()
    for script in SCRIPTS.iterdir():
        if script.is_file():
            (links / script.name).symlink_to(script)

    result = subprocess.run(
        [str(links / command), *args],
        cwd=tmp_path,
        env={**os.environ, "OPEN_DESIGN_REPO": str(tmp_path / "missing repository")},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == status, result.stderr
    assert expected in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("command", "args", "sibling", "expected"),
    [
        (
            "graphify-project-setup",
            [],
            "graphify-project-sync",
            ["--full", "--project"],
        ),
        (
            "graphify-update-and-commit",
            ["--no-report"],
            "graphify-project-sync",
            ["--update", "--project"],
        ),
        ("update-open-design", [], "open-design", ["--help"]),
    ],
)
def test_sibling_commands_resolve_from_relocated_script(tmp_path, command, args, sibling, expected):
    scripts = tmp_path / "relocated scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPTS / command, scripts / command)

    helper = scripts / sibling
    helper.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_CALL_LOG"\n', encoding="utf-8")
    helper.chmod(0o755)

    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("graphify", "corepack", "git"):
        tool = tools / name
        tool.write_text(
            '#!/bin/bash\n'
            'if [[ "$*" == *"rev-parse --show-toplevel"* ]]; then\n'
            '    printf "%s\\n" "$TEST_PROJECT"\n'
            'fi\n'
            'exit 0\n',
            encoding="utf-8",
        )
        tool.chmod(0o755)

    project = tmp_path / "test project"
    project.mkdir()
    link = tmp_path / command
    link.symlink_to(scripts / command)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        [str(link), *args],
        cwd=project,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{tools}:/usr/bin:/bin",
            "OPEN_DESIGN_REPO": str(project),
            "OPEN_DESIGN_DATA_DIR": str(tmp_path / "data"),
            "TEST_PROJECT": str(project),
            "TEST_CALL_LOG": str(log),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    if sibling == "graphify-project-sync":
        expected = [*expected, str(project), *args]
    assert log.read_text(encoding="utf-8").splitlines() == expected


def test_docker_cleanup_uses_external_docker_command(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_CALL_LOG"\n', encoding="utf-8")
    docker.chmod(0o755)
    log = tmp_path / "docker.log"

    subprocess.run(
        [str(SCRIPTS / "docker-clean.sh")],
        env={**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin", "TEST_CALL_LOG": str(log)},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    assert log.read_text(encoding="utf-8").splitlines() == [
        "system prune -a --volumes",
        "volume prune -a",
        "image prune -a",
    ]


@pytest.mark.parametrize(
    ("script", "dependencies"),
    [
        ("test_avi_to_mp4.sh", ("ffmpeg",)),
        ("test_gh_loc_chart.sh", ("jq",)),
    ],
)
def test_migrated_shell_regressions(script, dependencies):
    for dependency in dependencies:
        if not shutil.which(dependency):
            pytest.skip(f"{dependency} não está instalado")

    result = subprocess.run(
        ["bash", str(ROOT / "tests" / "integration" / script)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
