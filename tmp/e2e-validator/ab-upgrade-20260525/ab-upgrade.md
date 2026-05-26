# ab upgrade validation

## Stack/app state and URLs
- CLI-only validation in `/home/allanbatista/Apps/linux-utilities`.
- Python runtime: `rtk .venv/bin/python --version` -> exit `0`, stdout `Python 3.12.11`.
- `rtk`: `rtk which rtk` -> exit `0`, stdout `/home/allanbatista/.local/bin/rtk`.
- ShellCheck unavailable: `rtk shellcheck --version` -> exit `127`, stderr `[rtk: No such file or directory (os error 2)]`.
- No browser/app server URL applies.

## Commands run
- `rtk sed -n '1,220p' AGENTS.md`
- `rtk sed -n '1,220p' README.md`
- `rtk sed -n '1,220p' /home/allanbatista/.codex/skills/feature-workflow/SKILL.md`
- `rtk sed -n '1,160p' /home/allanbatista/.codex/skills/task-completion-report/SKILL.md`
- `rtk rg --files -g 'spec.md' -g 'plan.md' -g 'progress.md' .features`
- `rtk rg -n "upgrade|AC Traceability|Matriz AC|Acceptance|AC-" .features README.md AGENTS.md tests/integration/test_upgrade.py pyproject.toml`
- `rtk .venv/bin/python --version`
- `rtk which rtk`
- `rtk sed -n '25,32p;113,122p' .features/20260525-1640-ab-upgrade/spec.md .features/20260525-1640-ab-upgrade/plan.md`
- `rtk sed -n '664,678p' README.md`
- `rtk shellcheck --version`
- `rtk mkdir -p tmp/e2e-validator/ab-upgrade-20260525`
- `rtk bash bin/ab help`
- `rtk bash bin/ab upgrade --help`
- `rtk git diff --check`
- `rtk git status --short --untracked-files=all`
- `rtk bash bin/ab upgrade --dry-run`
- `rtk bash bin/ab upgrade`
- `rtk .venv/bin/python -m pytest tests/integration/test_upgrade.py -v`
- `rtk git status --short --untracked-files=all`

## Scenario results
- PASS AC-1 help: `rtk bash bin/ab help` -> exit `0`; stdout includes `upgrade           Update this ab installation`.
- PASS AC-2 command help: `rtk bash bin/ab upgrade --help` -> exit `0`; stdout includes `Usage: ab upgrade [--dry-run]`, `Runs non-interactively and never asks for input.`, `Preserves local changes; normal mode stops on a dirty worktree.`, and `Returns non-zero when any upgrade step fails.`
- PASS AC-4 dry-run: `rtk bash bin/ab upgrade --dry-run` -> exit `0`; stdout includes `Dry run: no changes will be made`, exact planned commands for `git fetch`, `git pull`, venv creation, pip upgrade/install, and `Dry run complete`.
- PASS AC-6 current dirty-worktree failure: `rtk bash bin/ab upgrade` -> exit `1`; stdout lists dirty paths and ends with `Error: worktree is not clean; commit, stash, or remove local changes before upgrading`.
- PASS AC-3/AC-5/AC-6 automated stubs: `rtk .venv/bin/python -m pytest tests/integration/test_upgrade.py -v` -> exit `0`; stdout reports `7 passed in 0.73s`, including `test_upgrade_runs_noninteractive_flow`, `test_upgrade_stops_on_failed_step`, and `test_upgrade_preserves_dirty_worktree`.
- PASS whitespace check: `rtk git diff --check` -> exit `0`; no stdout/stderr.

## Screenshot paths
- None; CLI-only validation.

## Persistence or side-effect checks
- `rtk git status --short --untracked-files=all` before and after validation showed the same tracked/untracked project paths for the dirty checkout.
- No real upgrade, fetch, pull, pip install, symlink, or completion mutation occurred; normal mode stopped on dirty worktree.

## Deviations from the requested journey
- None. ShellCheck was checked only to confirm unavailability.

## Final verdict
- PASS.
