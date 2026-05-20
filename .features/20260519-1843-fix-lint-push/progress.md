# Estado atual

Status: doing

Motivo: correção de `flake8 tests/` aplicada em arquivos de teste e validada localmente. CI verde em novo push ainda não foi observado, então a feature não está DONE.

Comando de sincronização usado: `rtk git status --short --untracked-files=all`

## Arquivos tocados

### Novos

- `.features/20260519-1843-fix-lint-push/spec.md` - documento da feature observado como untracked.
- `.features/20260519-1843-fix-lint-push/plan.md` - documento da feature observado como untracked.
- `.features/20260519-1843-fix-lint-push/progress.md` - criado por este controle.

### Modificados

- `tests/integration/test_auto_commit.py` - quebras de linhas longas em mocks/strings para `flake8 tests/`.
- `tests/unit/test_auto_commit.py` - remoção de import não usado.
- `tests/unit/test_git_helpers.py` - remoção de imports não usados.

### Removidos

- Nenhum.

## Validações registradas

- `rtk .venv/bin/python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__`: exit `0`.
- `rtk .venv/bin/python -m pytest tests/integration/test_auto_commit.py tests/unit/test_auto_commit.py tests/unit/test_git_helpers.py -q`: exit `0`, `43 passed in 0.90s`.
- `rtk .venv/bin/python -m flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero`: exit `0`, report-only output preserved by workflow contract.
- GitHub Actions failed run `26127058493`: `shellcheck` job exit success; `lint` job failed only on `flake8 tests/`.
- Contrato do workflow verificado por inspeção: `.github/workflows/lint.yml` contém `push branches: ['**']`, `flake8 tests/ --max-line-length=120 --exclude=__pycache__`, `flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero` e `ludeeus/action-shellcheck@master`.

# F1 - Diagnóstico

## F1.S1 - Reproduzir gates bloqueantes

### F1.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum por esta tarefa.
- Evidência requerida: saída e exit code de `python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__`.
- Evidência produzida: `rtk .venv/bin/python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__` retornou exit `0`.
- Bloqueador/causa: nenhum.

### F1.S1.T2

- Status: done
- Owner/subagent: executor ou subagent shell.
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum.
- Evidência requerida: violações atuais de `shellcheck --severity=warning bin/*` registradas.
- Evidência produzida: GitHub Actions failed run `26127058493` mostrou job `shellcheck` com `state: SUCCESS`.
- Bloqueador/causa: nenhum; não houve alteração em `bin/**`.

### F1.S1.T3

- Status: done
- Owner/subagent: executor ou subagent shell.
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum.
- Evidência requerida: violações atuais de `shellcheck --severity=warning scripts/*` registradas.
- Evidência produzida: GitHub Actions failed run `26127058493` mostrou job `shellcheck` com `state: SUCCESS`.
- Bloqueador/causa: nenhum; não houve alteração em `scripts/**`.

Validation Gate F1:

- Status: done
- Evidência produzida: `flake8 tests/` passou localmente; ShellCheck passou no GitHub Actions run `26127058493`.

# F2 - Correção mínima

## F2.S1 - Corrigir lint bloqueante

### F2.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: somente `tests/**` apontados por `F1.S1.T1`.
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`, `tests/unit/test_auto_commit.py`, `tests/unit/test_git_helpers.py`.
- Evidência requerida: `flake8 tests/` passa com exit code `0`.
- Evidência produzida: `rtk .venv/bin/python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__` retornou exit `0`.
- Bloqueador/causa: nenhum.

### F2.S1.T2

- Status: done
- Owner/subagent: executor ou subagent shell.
- Arquivos planejados: somente `bin/**` apontados por `F1.S1.T2`.
- Arquivos reais tocados: nenhum.
- Evidência requerida: `shellcheck --severity=warning bin/*` passa com exit code `0`.
- Evidência produzida: GitHub Actions failed run `26127058493` mostrou job `shellcheck` com `state: SUCCESS`.
- Bloqueador/causa: nenhum; não houve edição em `bin/**`.

### F2.S1.T3

- Status: done
- Owner/subagent: executor ou subagent shell.
- Arquivos planejados: somente `scripts/**` apontados por `F1.S1.T3`.
- Arquivos reais tocados: nenhum.
- Evidência requerida: `shellcheck --severity=warning scripts/*` passa com exit code `0`.
- Evidência produzida: GitHub Actions failed run `26127058493` mostrou job `shellcheck` com `state: SUCCESS`.
- Bloqueador/causa: nenhum; não houve edição em `scripts/**`.

Validation Gate F2:

- Status: done
- Evidência produzida: `flake8 tests/` passou localmente; ShellCheck passou no GitHub Actions run `26127058493`.

# F3 - Validação final

## F3.S1 - Confirmar contrato e regressão

### F3.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `.github/workflows/lint.yml` leitura.
- Arquivos reais tocados: nenhum.
- Evidência requerida: contrato comparado com spec.
- Evidência produzida: inspeção de `.github/workflows/lint.yml` confirmou `push branches: ['**']`, `flake8 tests/` bloqueante, `flake8 src/ ... --exit-zero` e ações ShellCheck.
- Bloqueador/causa: nenhum.

### F3.S1.T2

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/**` apenas se alterados.
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`, `tests/unit/test_auto_commit.py`, `tests/unit/test_git_helpers.py`.
- Evidência requerida: pytest relevante quando `tests/**` mudar.
- Evidência produzida: `rtk .venv/bin/python -m pytest tests/integration/test_auto_commit.py tests/unit/test_auto_commit.py tests/unit/test_git_helpers.py -q` retornou exit `0`, `43 passed in 0.90s`.
- Bloqueador/causa: nenhum.

### F3.S1.T3

- Status: todo
- Owner/subagent: e2e-validator
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum.
- Evidência requerida: aprovação ou bloqueios do e2e-validator registrados.
- Evidência produzida: nenhuma.
- Bloqueador/causa: validação e2e ainda não executada; ShellCheck também segue pendente.

Validation Gate F3:

- Status: doing
- Evidência produzida: `flake8 tests/`, pytest relevante e inspeção do workflow registrados.
- Evidência pendente: CI `Lint` verde em novo push.

# Pendências

- Registrar link do GitHub Actions `Lint` verde em novo push antes de marcar DONE.
