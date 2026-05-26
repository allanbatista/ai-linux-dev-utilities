# Status

READY_FOR_EXEC

## Approach

Criar `ab upgrade` como comando raiz Bash, sem LLM/API e sem novas dependencias. O comando resolve `PROJECT_DIR` pelo proprio `bin/ab-upgrade`, exige checkout git seguro, falha em worktree sujo no modo normal, executa update git fast-forward, refresca venv/dependencias e garante symlink/completion sem sobrescrever alvos divergentes. `--dry-run` imprime as acoes/comandos exatos para o estado atual e nao altera HEAD, indice, worktree, venv, symlinks ou completions.

A mudanca publica de CLI (`ab upgrade`) esta aprovada pela solicitacao do usuario.

## Stack real

Bash em `bin/`, Python src-layout para comandos Python existentes, pytest, flake8 e ShellCheck via CI. Nao ha template de plano em `.claude/`, `docs/` ou `.github/`.

## Interfaces / Contracts

| Interface | Contrato |
| --- | --- |
| CLI raiz | `ab upgrade [--dry-run] [-h|--help]` |
| Help raiz | `ab help` lista `upgrade` em comandos raiz com descricao curta |
| Help especifico | `ab upgrade --help` retorna `0` e documenta update normal, `--dry-run`, modo nao interativo, preservacao de alteracoes locais e falha non-zero |
| Normal | requer checkout git limpo; executa `git -C "$PROJECT_DIR" fetch --prune origin`, `git -C "$PROJECT_DIR" pull --ff-only`, cria `.venv` se ausente, roda pip upgrade/install, garante symlink `/usr/local/bin/ab` e completion do usuario sem prompt |
| Dry-run | imprime comandos/acoes que seriam executados para o estado atual; retorna `0`; nao escreve nada |
| Falhas | interrompe na primeira etapa falha, imprime etapa e retorna non-zero |
| Completion | nivel 1 inclui `upgrade`; `ab upgrade` completa `--dry-run -h --help` |
| Config/API | nenhuma mudanca em `~/.ab/config.json`, env vars, OpenRouter, schemas ou contratos externos |

## Technical Inventory / Inventario Tecnico

| slug/id | artefato/componente | query/filtros/url state | output type | dataset/permission gate | retailer/industry compat | teste alvo |
| --- | --- | --- | --- | --- | --- | --- |
| `root.upgrade` | `bin/ab-upgrade` | nenhuma query; flag `--dry-run` | stdout/stderr + exit code | git local, python3, pip, permissao opcional via `sudo -n` | nao aplicavel | `tests/integration/test_upgrade.py` |
| `root.help.upgrade` | `bin/ab` | nenhum filtro/url state | texto de help | nenhum | nao aplicavel | subprocess `bin/ab help` |
| `completion.upgrade` | `completions/ab.bash-completion` | shell words | sugestoes Bash | nenhum | nao aplicavel | teste shell via bash/completion |
| `docs.upgrade` | `README.md` | nenhum | docs Markdown | nenhum | nao aplicavel | revisao + grep |

## Arquivos afetados

Planejados:
- `bin/ab-upgrade` novo.
- `bin/ab` despacho/help raiz.
- `completions/ab.bash-completion` comandos/opcoes.
- `README.md` documentacao minima.
- `tests/integration/test_upgrade.py` testes do comando raiz.

Referencia somente leitura:
- `install.sh`, `tests/conftest.py`, `.github/workflows/tests.yml`, `.github/workflows/lint.yml`.

Preservar dirty work existente do spec: `README.md`, `bin/ab`, `completions/ab.bash-completion`, `src/ab_cli/core/config.py`, `src/ab_cli/utils/__init__.py`, `src/ab_cli/utils/api.py`, `tests/unit/test_config.py`, `.features/20260522-1130-media-audio-transcription/`, `bin/ab-media`, `src/ab_cli/commands/media.py`, `tests/integration/test_media.py`, `tests/unit/test_media.py`.

## Phases / Task Breakdown

### F1 - Contrato e comando

| ID | Owner | Paralelo | Arquivos | Dep | Tarefa | Done/evidencia |
| --- | --- | --- | --- | --- | --- | --- |
| F1.S1.T1 | executor | nao | nenhum | - | Capturar baseline com `rtk git status --short --untracked-files=all` e diffs dos arquivos dirty que serao tocados. | Evidencia registrada no handoff; nenhuma alteracao revertida. |
| F1.S1.T2 | executor | nao | `bin/ab-upgrade` | F1.S1.T1 | Criar script Bash POSIX-friendly com help, parser `--dry-run`, preflight git/python, clean-worktree normal, execucao por etapas e reporter de falha. | `bin/ab-upgrade --help` e `bin/ab-upgrade --dry-run` funcionam; ShellCheck sem warnings novos. |
| F1.S2.T1 | executor | nao | `bin/ab` | F1.S1.T2 | Adicionar `upgrade` no help raiz e no `case`, executando `"$SCRIPT_DIR/ab-upgrade" "$@"`. | `bin/ab help` lista `upgrade`; `bin/ab upgrade --help` retorna `0`. |

Validation Gate F1:
- `rtk bash bin/ab-upgrade --help`
- `rtk bash bin/ab-upgrade --dry-run`
- `rtk bash bin/ab help`
- `rtk bash bin/ab upgrade --help`
- `rtk shellcheck bin/ab bin/ab-upgrade` se ShellCheck estiver disponivel.

### F2 - Completion e docs

| ID | Owner | Paralelo | Arquivos | Dep | Tarefa | Done/evidencia |
| --- | --- | --- | --- | --- | --- | --- |
| F2.S1.T1 | executor | sim | `completions/ab.bash-completion` | F1.S2.T1 | Incluir `upgrade` em `level1` e `upgrade_opts="--dry-run -h --help"` para root command. | Bash completion sugere `upgrade` no nivel 1 e opcoes no nivel 2. |
| F2.S1.T2 | docs | sim | `README.md` | F1.S2.T1 | Documentar `ab upgrade`, `--dry-run`, seguranca de worktree, non-interactive e exemplos. | README tem secao/comando e nao remove docs dirty existentes. |

Validation Gate F2:
- `rtk rg -n "upgrade|--dry-run" README.md completions/ab.bash-completion`
- `rtk bash -n completions/ab.bash-completion`
- Teste manual de completion via `bash -lc` com `_ab_completions` carregado e evidencia das sugestoes.

### F3 - Testes

| ID | Owner | Paralelo | Arquivos | Dep | Tarefa | Done/evidencia |
| --- | --- | --- | --- | --- | --- | --- |
| F3.S1.T1 | test | sim | `tests/integration/test_upgrade.py` | F1.S2.T1 | Testar `ab help` e `ab upgrade --help` via subprocess. | AC-1/AC-2 cobertos. |
| F3.S2.T1 | test | sim | `tests/integration/test_upgrade.py` | F1.S1.T2 | Testar fluxo normal em repo temporario com stubs de `git`, `python3`, `pip`/`sudo`, validando ordem e argumentos. | AC-3 coberto sem rede/sudo real. |
| F3.S2.T2 | test | sim | `tests/integration/test_upgrade.py` | F1.S1.T2 | Testar `--dry-run` com HEAD/indice/worktree/symlink/completion antes/depois. | AC-4 coberto. |
| F3.S3.T1 | test | sim | `tests/integration/test_upgrade.py` | F1.S1.T2 | Injetar falha por etapa e validar interrupcao, mensagem da etapa e exit non-zero. | AC-5 coberto. |
| F3.S3.T2 | test | sim | `tests/integration/test_upgrade.py` | F1.S1.T2 | Testar worktree sujo normal: falha clara, paths preservados e sem git pull/pip/symlink. | AC-6 coberto. |

Validation Gate F3:
- `rtk python -m pytest tests/integration/test_upgrade.py -v`
- `rtk flake8 tests/integration/test_upgrade.py --max-line-length=120 --exclude=__pycache__`

### F4 - Validacao final e handoff

| ID | Owner | Paralelo | Arquivos | Dep | Tarefa | Done/evidencia |
| --- | --- | --- | --- | --- | --- | --- |
| F4.S1.T1 | executor | nao | todos planejados | F1-F3 | Rodar gates completos locais e revisar diff para conter somente escopo. | Logs dos comandos e `git diff --check`. |
| F4.S1.T2 | e2e-validator | nao | app local | F4.S1.T1 | Validar CLI real com repo temporario: help, dry-run, sucesso stubado, falha stubada, dirty worktree. | Relatorio/evidencia do `e2e-validator`. |

Validation Gate F4:
- `rtk python -m pytest tests/ -v`
- `rtk flake8 tests/ --max-line-length=120 --exclude=__pycache__`
- `rtk flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero`
- `rtk git diff --check`
- `rtk shellcheck bin/ab bin/ab-upgrade bin/ab-git bin/ab-util completions/ab.bash-completion scripts/passgenerator` se disponivel.
- Handoff `e2e-validator`: executar cenarios de F4.S1.T2 e anexar stdout/stderr/exit codes.

## Test Strategy

Criar testes de integracao com repo temporario e comandos stubados no `PATH` para evitar rede, sudo, pip real e prompts. Validar saida, exit code, ordem dos comandos, ausencia de mutacao em dry-run, bloqueio de dirty worktree e falha por etapa. Nao ha teste unitario novo porque a implementacao planejada e Bash sem novo util Python.

## AC Traceability / Matriz AC

| AC | Tarefas | Evidencia de validacao |
| --- | --- | --- |
| AC-1 | F1.S2.T1, F3.S1.T1 | `rtk bash bin/ab help`; pytest confirma `upgrade` e descricao curta. |
| AC-2 | F1.S1.T2, F3.S1.T1 | `rtk bash bin/ab upgrade --help`; pytest confirma docs de normal, `--dry-run`, non-interactive, preservacao e non-zero. |
| AC-3 | F1.S1.T2, F3.S2.T1 | pytest com stubs confirma git fetch/pull fast-forward, venv/pip refresh, symlink/completion seguros, sem input e exit `0`. |
| AC-4 | F1.S1.T2, F3.S2.T2 | pytest confirma comandos/acoes impressos e HEAD/indice/worktree/venv/symlinks/completions inalterados. |
| AC-5 | F1.S1.T2, F3.S3.T1 | pytest injeta falha e confirma etapa reportada, interrupcao e exit non-zero. |
| AC-6 | F1.S1.T2, F3.S3.T2 | pytest confirma dirty worktree preservado antes/depois e bloqueio claro sem update. |

## Riscos & Rollback

- Worktree dirty bloqueia upgrade normal: comportamento esperado; rollback nao necessario porque nenhuma mutacao deve ocorrer antes do bloqueio.
- `git pull --ff-only` pode atualizar codigo e etapa posterior falhar: comando para e reporta; rollback manual via git fica a cargo do usuario, sem reset automatico.
- Symlink `/usr/local/bin/ab` pode exigir privilegio: usar `sudo -n`; se falhar, reportar etapa e nao pedir input.
- Arquivos ja sujos (`README.md`, `bin/ab`, completion) podem conter trabalho paralelo: aplicar patches pequenos; rollback por commit/tarefa com `git revert` ou patch inverso, nunca `reset --hard`.

## Out of Scope

Migracoes, telemetry, alteracao de config `~/.ab/config.json`, LLM/OpenRouter, version manager, rollback automatico, resolver dirty work existente, mudancas em `ab git`, `ab util`, `ab media` ou comandos Python.

## Paralelizacao / Subagents

- F2.S1.T1 e F2.S1.T2 podem rodar em paralelo apos F1.S2.T1.
- F3.S1-T3 podem ser escritos em paralelo apos congelar o contrato de F1.
- `e2e-validator` entra somente em F4.S1.T2, depois dos testes automatizados.
- Owner boundaries: executor altera Bash/docs/completion; test owner altera apenas `tests/integration/test_upgrade.py`; docs owner altera apenas README.

## Gate Final

Antes de entregar:
- Todos os ACs mapeados acima passam.
- `rtk python -m pytest tests/ -v` passa.
- Lint/test gates reais do repo passam ou falhas preexistentes sao documentadas com evidencia.
- `e2e-validator` valida help, dry-run, sucesso, falha e dirty worktree.
- `rtk git status --short --untracked-files=all` mostra somente mudancas esperadas e dirty preexistente preservado.

## Definition of Done

- `ab upgrade` esta disponivel como comando raiz, documentado e completavel.
- Normal, dry-run, falha e dirty worktree tem comportamento observavel por stdout/stderr, exit code e estado git/filesystem.
- Testes de integracao cobrem todos os ACs sem rede, sudo real ou LLM.
- Nenhum arquivo fora do escopo foi revertido, sobrescrito ou limpo.
