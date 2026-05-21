# Estado atual

Status: done

Motivo: implementação concluída e validada. O `auto-commit` mantém binários staged/commitáveis, mas remove caminhos/status/conteúdo binário do contexto enviado ao LLM.

Comando de sincronização usado: `rtk git status --short --untracked-files=all`

Bloqueios atuais: nenhum.

## Arquivos tocados

### Novos

- `.features/20260520-2201-auto-commit-ignore-binaries/spec.md`
- `.features/20260520-2201-auto-commit-ignore-binaries/plan.md`
- `.features/20260520-2201-auto-commit-ignore-binaries/progress.md`

### Modificados

- `src/ab_cli/utils/git_helpers.py`
- `src/ab_cli/utils/__init__.py`
- `src/ab_cli/commands/auto_commit.py`
- `tests/unit/test_git_helpers.py`
- `tests/integration/test_auto_commit.py`

### Removidos

- Nenhum.

## Validações registradas

- `rtk .venv/bin/python -m pytest tests/unit/test_git_helpers.py -v`: 8 passed.
- `rtk .venv/bin/python -m pytest tests/integration/test_auto_commit.py -v`: 37 passed.
- `rtk .venv/bin/python -m pytest tests/ -v`: 560 passed.
- `rtk python -m pytest ...`: bloqueado porque `/usr/bin/python` não tem `pytest`; validado com `.venv/bin/python`.

# F1 - Helpers text-only staged

## F1.S1 - Inventário staged via Git

### F1.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `src/ab_cli/utils/git_helpers.py`, `src/ab_cli/utils/__init__.py`
- Arquivos reais tocados: `src/ab_cli/utils/git_helpers.py`, `src/ab_cli/utils/__init__.py`
- Evidência requerida: `get_staged_text_files()` retorna `list[str]` sem caminhos binários.
- Evidência produzida: `TestStagedTextFiles::test_get_staged_text_files_excludes_binary` e `test_get_staged_text_files_only_binary_empty` passaram.
- Bloqueador/causa: nenhum.

### F1.S1.T2

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `src/ab_cli/utils/git_helpers.py`, `src/ab_cli/utils/__init__.py`
- Arquivos reais tocados: `src/ab_cli/utils/git_helpers.py`, `src/ab_cli/utils/__init__.py`
- Evidência requerida: helpers de diff/status retornam `""` com lista vazia e limitam saída aos arquivos informados.
- Evidência produzida: `TestStagedTextFiles::test_get_staged_diff_and_status_for_files_limit_pathspec` passou.
- Bloqueador/causa: nenhum.

### F1.S1.T3

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/unit/test_git_helpers.py`
- Arquivos reais tocados: `tests/unit/test_git_helpers.py`
- Evidência requerida: testes com repo real para mixed text+binary staged, only binary staged e pathspec text-only.
- Evidência produzida: `rtk .venv/bin/python -m pytest tests/unit/test_git_helpers.py -v`: 8 passed.
- Bloqueador/causa: nenhum.

Validation Gate F1:

- Status: done
- Comando requerido: `rtk python -m pytest tests/unit/test_git_helpers.py -v`
- Evidência produzida: `/usr/bin/python` sem `pytest`; equivalente no venv passou com 8 tests.
- Bloqueador/causa: nenhum.

# F2 - Integração no auto-commit

## F2.S1 - Contexto LLM filtrado

### F2.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `src/ab_cli/commands/auto_commit.py`
- Arquivos reais tocados: `src/ab_cli/commands/auto_commit.py`
- Evidência requerida: `generate_commit_plan()` recebe somente `diff` e `name_status` textuais.
- Evidência produzida: testes de integração capturam `prompt_text` sem `asset.bin`, `A\tasset.bin` e `binary-secret`.
- Bloqueador/causa: nenhum.

### F2.S1.T2

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `src/ab_cli/commands/auto_commit.py`
- Arquivos reais tocados: `src/ab_cli/commands/auto_commit.py`
- Evidência requerida: only-binary staged emite aviso, não chama LLM e não cria commit.
- Evidência produzida: `test_main_only_binary_staged_skips_llm_and_commit` passou.
- Bloqueador/causa: nenhum.

### F2.S1.T3

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `src/ab_cli/commands/auto_commit.py`
- Arquivos reais tocados: `src/ab_cli/commands/auto_commit.py`
- Evidência requerida: resumo local e `stage_all_files()` preservados; binários continuam staged/commitáveis quando houver texto.
- Evidência produzida: testes mixed e `-y -Y` confirmam commit contendo `asset.bin`.
- Bloqueador/causa: nenhum.

Validation Gate F2:

- Status: done
- Comando requerido: `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- Evidência produzida: equivalente no venv passou com 37 tests.
- Bloqueador/causa: nenhum.

# F3 - Cobertura dos critérios de aceite

## F3.S1 - Testes de fluxo CLI

### F3.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/integration/test_auto_commit.py`
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`
- Evidência produzida: `test_main_excludes_staged_binary_from_prompt_but_commits_it` passou.
- Bloqueador/causa: nenhum.

### F3.S1.T2

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/integration/test_auto_commit.py`
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`
- Evidência produzida: `test_main_add_excludes_binary_from_prompt_but_stages_it` passou.
- Bloqueador/causa: nenhum.

### F3.S1.T3

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/integration/test_auto_commit.py`
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`
- Evidência produzida: `test_main_staged_only_excludes_staged_binary` passou.
- Bloqueador/causa: nenhum.

### F3.S1.T4

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/integration/test_auto_commit.py`
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`
- Evidência produzida: `test_main_only_binary_staged_skips_llm_and_commit` passou.
- Bloqueador/causa: nenhum.

### F3.S1.T5

- Status: done
- Owner/subagent: executor
- Arquivos planejados: `tests/integration/test_auto_commit.py`
- Arquivos reais tocados: `tests/integration/test_auto_commit.py`
- Evidência produzida: asserts negativos para `asset.bin`, `A\tasset.bin` e `binary-secret` nos testes novos.
- Bloqueador/causa: nenhum.

Validation Gate F3:

- Status: done
- Comando requerido: `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- Evidência produzida: equivalente no venv passou com 37 tests.
- Bloqueador/causa: nenhum.

# F4 - Validação final

## F4.S1 - Gates do repositório

### F4.S1.T1

- Status: done
- Owner/subagent: executor
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum.
- Evidência produzida: `rtk .venv/bin/python -m pytest tests/unit/test_git_helpers.py -v`: 8 passed.
- Bloqueador/causa: nenhum.

### F4.S1.T2

- Status: done
- Owner/subagent: executor
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum.
- Evidência produzida: `rtk .venv/bin/python -m pytest tests/integration/test_auto_commit.py -v`: 37 passed.
- Bloqueador/causa: nenhum.

### F4.S1.T3

- Status: done
- Owner/subagent: executor
- Arquivos planejados: nenhum.
- Arquivos reais tocados: nenhum.
- Evidência produzida: `rtk .venv/bin/python -m pytest tests/ -v`: 560 passed.
- Bloqueador/causa: nenhum.

### F4.S1.T4

- Status: done
- Owner/subagent: executor
- Arquivos planejados: arquivos alterados e evidências dos comandos.
- Arquivos reais tocados: nenhum.
- Evidência produzida: testes automatizados validaram AC-1 a AC-7 sem rede/secrets.
- Bloqueador/causa: nenhum.

Validation Gate F4:

- Status: done
- Evidência produzida: todos os gates passaram no venv.
- Bloqueador/causa: nenhum.

# Pendências

- Nenhuma.
