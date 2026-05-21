# Status

READY_FOR_EXEC

# Approach

Filtrar o contexto usado por `ab git auto-commit` antes de chamar o LLM: manter staging/commit/push/PR exatamente como hoje, mas montar `diff` e `name_status` apenas com arquivos staged classificados pelo Git como textuais. Arquivos binários podem continuar aparecendo no resumo local de mudanças e continuar staged, mas seus caminhos/status/conteúdo não entram em `generate_commit_plan()`.

A decisão técnica é usar o índice do Git, não leitura do working tree, para classificar arquivos staged: `git diff --cached --numstat -z` marca binários com adições/deleções `-`; qualquer entrada com contadores numéricos é textual. Isso cobre arquivos adicionados, modificados e deletados no índice sem depender de extensão ou arquivo existir no disco.

Seguir padrões de [src/ab_cli/commands/auto_commit.py](/home/allanbatista/Apps/linux-utilities/src/ab_cli/commands/auto_commit.py), [src/ab_cli/utils/git_helpers.py](/home/allanbatista/Apps/linux-utilities/src/ab_cli/utils/git_helpers.py), [src/ab_cli/utils/__init__.py](/home/allanbatista/Apps/linux-utilities/src/ab_cli/utils/__init__.py), [tests/integration/test_auto_commit.py](/home/allanbatista/Apps/linux-utilities/tests/integration/test_auto_commit.py) e [tests/unit/test_git_helpers.py](/home/allanbatista/Apps/linux-utilities/tests/unit/test_git_helpers.py).

## Interfaces / Contracts

- CLI pública: sem novas flags, sem mudança de nomes, sem config/env nova.
- Contrato preservado: `-y/-a` continua stageando todos os arquivos; `-s` continua usando somente arquivos já staged; commit/push/PR/proteção de branch seguem o fluxo atual.
- Contrato alterado internamente: `generate_commit_plan(diff, name_status, ...)` passa a receber apenas diff/status textual staged.
- Novo contrato interno sugerido em `ab_cli.utils.git_helpers`:
  - `get_staged_text_files() -> list[str]`: retorna caminhos staged textuais, excluindo binários via `git diff --cached --numstat -z`.
  - `get_staged_diff_for_files(files: list[str]) -> str`: retorna `git diff --cached -- <files>`.
  - `get_staged_name_status_for_files(files: list[str]) -> str`: retorna `git diff --cached --name-status -- <files>`.
- Sem mudanças em API externa, JSON do LLM, persistência, autenticação, billing, infra ou dependências runtime.

## Technical Inventory / Inventário Técnico

Não é feature de dashboard/report/data.

- Slugs/queries/components/frontend: não aplicável.
- Retailer/industry compatibility expectation: não aplicável; CLI local sem variação por retailer/industry.
- Output types: mensagens de terminal existentes; novo aviso observável para ausência de mudanças textuais staged.
- Dataset/permission gate: permissões locais Git existentes.
- Comando: `ab git auto-commit`.
- Flags cobertas: default, `-y/-a`, `-Y`, `-s`, `-f`, `-p`, `-P`.
- Fonte de verdade para staging: índice Git após eventual `stage_all_files()`.
- Classificação binária: `git diff --cached --numstat -z`, entradas com `-\t-` são binárias e excluídas do contexto.
- Payload LLM: `prompt_text` passado para `call_llm_with_model_info()` dentro de `generate_commit_plan()`.
- Caminhos binários: não podem aparecer em `FILES CHANGED` nem em `DIFF` do prompt.

# Affected Files

- `src/ab_cli/utils/git_helpers.py`
- `src/ab_cli/utils/__init__.py`
- `src/ab_cli/commands/auto_commit.py`
- `tests/unit/test_git_helpers.py`
- `tests/integration/test_auto_commit.py`

# Phases / Task Breakdown

## F1 - Helpers text-only staged

### F1.S1 - Inventário staged via Git

- `F1.S1.T1` Owner: executor. Arquivos: `src/ab_cli/utils/git_helpers.py`, `src/ab_cli/utils/__init__.py`. Dependências: nenhuma. Fazer: adicionar/exportar `get_staged_text_files()` usando `git diff --cached --numstat -z`; parsear registros NUL; incluir apenas entradas com adições/deleções numéricas; excluir entradas `-\t-`; preservar ordem do Git. Done when: retorna `list[str]` sem caminhos binários.
- `F1.S1.T2` Owner: executor. Arquivos: `src/ab_cli/utils/git_helpers.py`, `src/ab_cli/utils/__init__.py`. Dependências: `F1.S1.T1`. Fazer: adicionar/exportar `get_staged_diff_for_files(files)` e `get_staged_name_status_for_files(files)`; se `files` vazio, retornar `""`; usar `run_git("diff", "--cached", "--", *files)` e `run_git("diff", "--cached", "--name-status", "--", *files)`. Done when: pathspec limita saída só aos arquivos informados.
- `F1.S1.T3` Owner: executor. Arquivos: `tests/unit/test_git_helpers.py`. Dependências: `F1.S1.T1`, `F1.S1.T2`. Fazer: adicionar testes unitários/integrados pequenos com repo real para mixed text+binary staged, only binary staged e pathspec text-only. Done when: helpers não retornam caminho binário e retornam diff/status textual.

Validation Gate F1:

- Comando: `rtk python -m pytest tests/unit/test_git_helpers.py -v`
- Evidência: testes novos de helpers passando e demonstrando exclusão de binários antes do fluxo CLI.
- Handoff e2e-validator: não obrigatório nesta fase, salvo falha intermitente de parsing Git.

## F2 - Integração no auto-commit

### F2.S1 - Contexto LLM filtrado

- `F2.S1.T1` Owner: executor. Arquivos: `src/ab_cli/commands/auto_commit.py`. Dependências: `F1.S1.T1`, `F1.S1.T2`. Fazer: importar os novos helpers e substituir `get_staged_diff()`/`get_staged_name_status()` no caminho de geração LLM por `text_files = get_staged_text_files()`, `diff = get_staged_diff_for_files(text_files)`, `name_status = get_staged_name_status_for_files(text_files)`. Done when: `generate_commit_plan()` só recebe contexto textual.
- `F2.S1.T2` Owner: executor. Arquivos: `src/ab_cli/commands/auto_commit.py`. Dependências: `F2.S1.T1`. Fazer: quando houver staged changes mas `text_files` ou `diff` textual estiver vazio, emitir aviso claro como `No staged text changes to generate commit message` e sair antes de chamar LLM ou criar commit. Done when: only-binary staged retorna sem chamada LLM e sem commit.
- `F2.S1.T3` Owner: executor. Arquivos: `src/ab_cli/commands/auto_commit.py`. Dependências: `F2.S1.T1`. Fazer: preservar resumo de mudanças baseado em `get_staged_files()`, `get_unstaged_files()`, `get_untracked_files()` e preservar `stage_all_files()` em `-y/-a`. Done when: binários continuam staged e commitáveis quando existe ao menos um arquivo textual.

Validation Gate F2:

- Comando: `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- Evidência: testes existentes de branch, push, PR e staged-only continuam passando após filtro textual.
- Handoff e2e-validator: revisar se o prompt capturado em testes não contém caminho/status/conteúdo binário.

## F3 - Cobertura dos critérios de aceite

### F3.S1 - Testes de fluxo CLI

- `F3.S1.T1` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: F2. Fazer: teste mixed staged text+binary com `-Y`; capturar `mock_call.call_args.args[0]`; validar presença do texto e caminho textual, ausência de caminho binário, status binário e conteúdo binário; validar commit inclui ambos os arquivos staged após confirmação automática. Cobre AC-1, AC-5, AC-6.
- `F3.S1.T2` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: F2. Fazer: teste `-y -Y` com arquivos textuais e binários não staged; validar `stage_all_files()` real ou spy, prompt só textual, e binário ainda staged/commitado. Cobre AC-2, AC-5, AC-7.
- `F3.S1.T3` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: F2. Fazer: teste `-s -Y` com text+binary staged e arquivos unstaged/untracked adicionais; validar prompt só contém texto staged, não contém binário staged nem arquivos não staged. Cobre AC-3, AC-7.
- `F3.S1.T4` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: F2. Fazer: teste only-binary staged; mockar `call_llm_with_model_info` para falhar se chamado; validar aviso observável, exit sem commit, e staged binary permanece staged. Cobre AC-4, AC-7.
- `F3.S1.T5` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: F2. Fazer: reforçar asserts negativos nos testes novos para strings exatas: nome binário (`asset.bin`), linha de status (`A\tasset.bin` quando aplicável) e bytes/conteúdo marcador (`PNG`/`binary-secret`). Done when: AC-6 tem evidência automatizada explícita.

Validation Gate F3:

- Comando: `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- Evidência: nomes dos testes e asserts de prompt/payload demonstram AC-1 a AC-7.
- Handoff e2e-validator: validar evidências dos prompts capturados e efeitos Git staged/commit.

## F4 - Validação final

### F4.S1 - Gates do repositório

- `F4.S1.T1` Owner: executor. Arquivos: nenhum. Dependências: F1-F3. Rodar `rtk python -m pytest tests/unit/test_git_helpers.py -v`. Done when: passa.
- `F4.S1.T2` Owner: executor. Arquivos: nenhum. Dependências: F1-F3. Rodar `rtk python -m pytest tests/integration/test_auto_commit.py -v`. Done when: passa.
- `F4.S1.T3` Owner: executor. Arquivos: nenhum. Dependências: F1-F3. Rodar `rtk python -m pytest tests/ -v`. Done when: passa.
- `F4.S1.T4` Owner: e2e-validator. Arquivos: arquivos alterados e evidências dos comandos. Dependências: `F4.S1.T1`-`F4.S1.T3`. Validar em repo temporário ou por revisão das evidências que mixed, only-binary, `-y -Y` e `-s -Y` cumprem os ACs sem rede/secrets. Done when: handoff registra ACs validados.

Validation Gate F4:

- Comandos: `rtk python -m pytest tests/unit/test_git_helpers.py -v`; `rtk python -m pytest tests/integration/test_auto_commit.py -v`; `rtk python -m pytest tests/ -v`
- Evidência: logs de pytest e nota do `e2e-validator`.

## AC Traceability / Matriz AC

| AC | Tasks | Evidência |
| --- | --- | --- |
| AC-1 | `F1.S1.T1`, `F1.S1.T2`, `F2.S1.T1`, `F3.S1.T1` | Teste mixed staged captura prompt e confirma diff/status textual presentes e caminho/conteúdo/status binário ausentes. |
| AC-2 | `F2.S1.T3`, `F3.S1.T2` | Teste `-y -Y` confirma staging atual preservado e prompt só textual após `stage_all_files()`. |
| AC-3 | `F2.S1.T1`, `F2.S1.T3`, `F3.S1.T3` | Teste `-s -Y` confirma uso exclusivo de textuais staged, sem binários staged e sem unstaged/untracked. |
| AC-4 | `F2.S1.T2`, `F3.S1.T4` | Teste only-binary staged valida aviso, ausência de chamada LLM e ausência de commit. |
| AC-5 | `F2.S1.T3`, `F3.S1.T1`, `F3.S1.T2` | Testes mixed e `-y -Y` validam que binários permanecem staged e entram no commit quando há texto válido. |
| AC-6 | `F2.S1.T1`, `F3.S1.T1`, `F3.S1.T5` | Asserts negativos no `prompt_text` para caminho, status e conteúdo binário; resumo local pode permanecer sem validação restritiva. |
| AC-7 | `F1.S1.T3`, `F3.S1.T1`-`F3.S1.T4`, `F4.S1.T2` | Suite de integração cobre mixed, only-binary, `-y -Y` e `-s -Y`; helper tests cobrem classificação. |

# Test Strategy

- Usar pytest existente, sem rede e sem OpenRouter real.
- Mockar `ab_cli.commands.auto_commit.call_llm_with_model_info` e inspecionar `prompt_text`.
- Usar git real no fixture `mock_git_repo` para staging/commit.
- Criar binários com bytes contendo marcador pesquisável, por exemplo `b"\x00PNG-binary-secret\xff"`, e validar que `asset.bin`, `A\tasset.bin` e `binary-secret` não aparecem no prompt.
- Para only-binary, mockar LLM com `side_effect=AssertionError("LLM should not be called")`.
- Para commits que devem incluir binário, validar `git show --name-only --format= HEAD` ou `git diff-tree --no-commit-id --name-only -r HEAD`.

# Risks & Rollback

- Risco: parsing de `--numstat -z` para renames pode variar. Mitigação: preservar testes de casos simples e implementar parser tolerante; se necessário, usar fallback por arquivo com `git diff --cached --numstat -- <path>`.
- Risco: arquivos deletados não existem no working tree. Mitigação: classificação deve vir do índice Git, não de `binaryornot` no caminho local.
- Risco: pathspec com nomes especiais. Mitigação: chamar `run_git(..., "--", *files)` sem shell.
- Rollback: reverter os commits dos helpers e da integração em `auto_commit` restaura o comportamento anterior; testes novos devem ser revertidos no mesmo rollback.

# Out of Scope

- Alterar `ab prompt`, `ab git pr-description`, `ab git changelog` ou outros comandos.
- Novas flags/configs para permitir binários no contexto.
- Mudanças no JSON esperado do LLM.
- Documentação de usuário, completions, instalação, modelos, autenticação ou CI.
- Bloquear commit de arquivos binários.

# Paralelização / Subagents

- Serial: `F1.S1.T1` antes de qualquer integração.
- Paralelizável depois de `F1.S1.T2`: `F1.S1.T3` e `F2.S1.T1` podem avançar por owners distintos se coordenarem exports em `utils/__init__.py`.
- Paralelizável depois de F2: `F3.S1.T1`, `F3.S1.T2`, `F3.S1.T3` e `F3.S1.T4` podem ser escritos em paralelo, cada um editando testes separados no mesmo arquivo com cuidado contra conflitos.
- Subagents úteis: executor principal para helpers/integração; subagent de testes para `tests/integration/test_auto_commit.py`; `e2e-validator` obrigatório no gate final.

# Gate Final

- `rtk python -m pytest tests/unit/test_git_helpers.py -v`
- `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- `rtk python -m pytest tests/ -v`
- `e2e-validator`: validar AC-1 a AC-7, confirmar que não houve chamada de rede nos testes e que nenhum caminho/status/conteúdo binário aparece no prompt capturado.

# Definition of Done

- `plan.md` está `READY_FOR_EXEC`.
- Tarefas têm IDs estáveis, owner, arquivos, dependências, done-when e evidência.
- Todos os ACs têm mapeamento explícito para tarefas e validação.
- Implementação futura não precisa escolher como classificar binários, quais arquivos tocar ou quais testes criar.
- Nenhuma alteração de código de implementação foi feita nesta etapa de planejamento.
