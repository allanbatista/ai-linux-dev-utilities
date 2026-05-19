# Status

READY_FOR_EXEC

# Approach

Alterar apenas o fluxo de `ab git auto-commit` quando `-y -Y -p -P` roda em `master` ou `main`: depois de gerar o plano LLM e antes do commit, se a branch atual for protegida, não houver `--force`, e existir `branch_name` sugerida, criar automaticamente essa branch com `create_branch(branch_name)`, atualizar `current_branch` e seguir com commit, push e PR. Manter o fluxo interativo atual para casos sem `-y -Y -p -P`.

Basear-se nos padrões existentes de [src/ab_cli/commands/auto_commit.py](/home/allanbatista/Apps/linux-utilities/src/ab_cli/commands/auto_commit.py) e nos testes de [tests/integration/test_auto_commit.py](/home/allanbatista/Apps/linux-utilities/tests/integration/test_auto_commit.py).

## Interfaces / Contracts

- CLI existente: `ab git auto-commit -y -Y -p -P`.
- Sem novas flags, config, env vars, dependências, schema ou API externa.
- Contrato alterado: em `master`/`main`, com `-y -Y -p -P` e sem `-f`, a branch sugerida pelo plano LLM passa a ser criada automaticamente antes do commit.
- Contratos preservados: `-P` exige `-p`; `-f` não cria branch automaticamente; PR direto de branch protegida continua bloqueado.

## Technical Inventory / Inventário Técnico

Não é feature de dashboard/report/data. Inventário aplicável:

- Slugs: não aplicável; nenhum slug ou id persistido.
- Queries: não aplicável; nenhuma consulta SQL/API nova.
- Components: não aplicável; CLI sem componentes frontend.
- Output types: saída de terminal existente, erros via stderr e mensagens de sucesso existentes.
- Filters/url state: não aplicável; sem filtros, URL state ou query params.
- Dataset/permission gating: não aplicável; usa permissões git/gh locais existentes.
- Retailer/industry compatibility: não aplicável.
- Comando: `ab git auto-commit`.
- Flags: `-y/--add`, `-Y/--yes`, `-p/--push`, `-P/--pr`, `-f/--force`.
- Branches protegidas: `master`, `main`, via `is_protected_branch`.
- Branch sugerida: `plan["branch_name"]`, normalizada por `normalize_branch_name`.
- Commit: `create_commit(commit_msg)`.
- Push: `push_branch(current_branch)`.
- PR: `handle_pr_flow(current_branch, lang, False)` após push já executado no fluxo principal.
- Base do PR: `detect_base_branch()` dentro de `handle_pr_flow`, deve permanecer apontando para a branch protegida original nos cenários dos testes.

# Affected Files

- `src/ab_cli/commands/auto_commit.py`
- `tests/integration/test_auto_commit.py`

# Phases / Task Breakdown

## F1 - Fluxo automático em branch protegida

### F1.S1 - Implementação mínima

- `F1.S1.T1` Owner: executor. Arquivos: `src/ab_cli/commands/auto_commit.py`. Dependências: nenhuma. Fazer: detectar o modo automático completo com `args.add and args.yes_commit and args.push and args.pr and not args.force`; quando `on_protected_branch` for verdadeiro, criar `branch_name` com `create_branch`, falhar com `sys.exit(1)` se não houver sugestão ou criação falhar, atualizar `current_branch` e `on_protected_branch`. Done when: nenhum `input()` é chamado nesse modo e o commit usa a branch criada.
- `F1.S1.T2` Owner: executor. Arquivos: `src/ab_cli/commands/auto_commit.py`. Dependências: `F1.S1.T1`. Fazer: preservar o bloco interativo `handle_protected_branch` para todos os demais casos protegidos sem `--force`. Done when: testes existentes de fluxo interativo/force continuam válidos.
- `F1.S1.T3` Owner: executor. Arquivos: `src/ab_cli/commands/auto_commit.py`. Dependências: `F1.S1.T1`. Fazer: garantir que `-f -y -Y -p -P` continue chegando ao erro existente de `-P requires a non-protected branch`. Done when: não chama `create_branch`, `push_branch` ou `create_pr` nesse cenário.

Validation Gate F1:

- Comando: `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- Evidência: testes novos e existentes de `auto_commit` passando.
- Handoff: acionar `e2e-validator` se falhar comportamento real de CLI ou se houver divergência entre mocks e git local.

## F2 - Cobertura automatizada

### F2.S1 - Testes de cenários obrigatórios

- `F2.S1.T1` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: `F1.S1.T1`. Fazer: adicionar teste para `master` com mudanças staged, argv `["auto-commit", "-y", "-Y", "-p", "-P"]`, LLM retornando `feature/protected-master`, `create_branch` real ou spy, `push_branch` mockado, `create_pr` mockado. Validar branch final, commit na branch criada, push da branch criada, PR com base `master`, e ausência de prompt.
- `F2.S1.T2` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: `F1.S1.T1`. Fazer: adicionar teste equivalente para `main`; criar/checkout `main` no fixture antes das mudanças. Validar commit/push/PR a partir da branch sugerida e base `main`.
- `F2.S1.T3` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: nenhuma. Fazer: manter ou ajustar teste de branch não protegida para `-y -Y -p -P`, validando que não chama `create_branch`/`handle_protected_branch` e usa a branch atual.
- `F2.S1.T4` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: nenhuma. Fazer: adicionar/ajustar teste de `-f -y -Y -p -P` em branch protegida, validando exit `1`, mensagem `-P requires a non-protected branch`, sem branch automática, sem push, sem PR.
- `F2.S1.T5` Owner: executor. Arquivos: `tests/integration/test_auto_commit.py`. Dependências: nenhuma. Fazer: manter teste `-P` sem `-p` e reforçar que falha antes de commit/push/PR usando mocks se necessário.

Validation Gate F2:

- Comando: `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- Evidência: AC-8 coberto por nomes de testes explícitos e asserts de branch/push/PR.
- Handoff: `e2e-validator` revisa se todos os ACs têm evidência automatizada.

## F3 - Validação final

### F3.S1 - Gates do repositório

- `F3.S1.T1` Owner: executor. Arquivos: nenhum. Dependências: F1, F2. Rodar `rtk python -m pytest tests/integration/test_auto_commit.py -v`. Done when: passa.
- `F3.S1.T2` Owner: executor. Arquivos: nenhum. Dependências: F1, F2. Rodar `rtk python -m pytest tests/ -v`. Done when: passa.
- `F3.S1.T3` Owner: e2e-validator. Arquivos: implementação e testes alterados. Dependências: `F3.S1.T1`. Validar manualmente o contrato CLI em repo temporário ou revisar evidências de testes: `master/main` criam branch sugerida, commit/push/PR usam a branch criada, `-f` e `-P` sem `-p` falham corretamente. Done when: evidência anexada no handoff final.

Validation Gate F3:

- Comandos: `rtk python -m pytest tests/integration/test_auto_commit.py -v`; `rtk python -m pytest tests/ -v`
- Evidência: logs de pytest e nota do `e2e-validator`.

## AC Traceability / Matriz AC

| AC | Tasks | Evidência |
| --- | --- | --- |
| AC-1 | `F1.S1.T1`, `F2.S1.T1` | Evidência: teste em `master` confirma criação automática da branch sugerida e sem `input()`. |
| AC-2 | `F1.S1.T1`, `F2.S1.T2` | Evidência: teste em `main` confirma criação automática da branch sugerida e sem `input()`. |
| AC-3 | `F1.S1.T1`, `F2.S1.T1`, `F2.S1.T2` | Evidência: assert de branch atual/latest commit na branch sugerida, não na protegida. |
| AC-4 | `F1.S1.T1`, `F2.S1.T1`, `F2.S1.T2` | Evidência: asserts `push_branch("feature/...")` e `create_pr(..., base_branch)` com base `master`/`main`. |
| AC-5 | `F1.S1.T2`, `F2.S1.T3` | Evidência: teste de branch não protegida continua usando branch atual e não cria branch. |
| AC-6 | `F1.S1.T3`, `F2.S1.T4` | Evidência: teste `-f` em branch protegida falha antes de push/PR e não cria branch. |
| AC-7 | `F2.S1.T5` | Evidência: teste `-P` sem `-p` retorna exit `1` antes de efeitos colaterais. |
| AC-8 | `F2.S1.T1`-`F2.S1.T5`, `F3.S1.T1` | Evidência: suite `tests/integration/test_auto_commit.py` passando. |

# Test Strategy

- Prioridade: testes de integração em `tests/integration/test_auto_commit.py`, com LLM, push, gh e PR mockados.
- Usar git real do fixture para branch/commit quando possível.
- Mockar `builtins.input` para lançar `AssertionError` nos cenários não interativos.
- Evitar rede e GitHub real: `push_branch`, `check_gh_installed`, `check_gh_authenticated`, `generate_pr_content`, `create_pr` mockados.

# Risks & Rollback

- Risco: `detect_base_branch()` pode escolher base errada se `main` e `master` coexistirem no fixture. Mitigação: preparar cada teste com apenas a base esperada ou mockar `detect_base_branch` quando o objetivo for fluxo de auto-branch.
- Risco: branch sugerida vazia em modo automático. Mitigação: falhar com erro claro antes do commit.
- Rollback: reverter o commit de `F1.S1.T1` restaura o prompt protegido anterior; testes novos devem ser revertidos junto.

# Out of Scope

- Novas flags.
- Mudanças em `ab git pr-description`, completions, README, install ou configuração.
- Integração real com GitHub/gh em testes automatizados.
- Refatoração ampla de helpers git ou de geração LLM.

# Paralelização / Subagents

- Paralelizável após `F1.S1.T1`: `F2.S1.T1` e `F2.S1.T2` podem ser escritos em paralelo por subagents de teste, desde que não editem o mesmo bloco simultaneamente.
- Paralelizável sem dependência: `F2.S1.T4` e `F2.S1.T5`.
- Serial: `F3` só depois de F1/F2.
- Subagents úteis: executor principal para implementação; `e2e-validator` para gate final.

# Gate Final

- `rtk python -m pytest tests/integration/test_auto_commit.py -v`
- `rtk python -m pytest tests/ -v`
- `e2e-validator`: validar evidência de AC-1 a AC-8 e confirmar ausência de rede/secrets nos testes.

# Definition of Done

- `plan.md` está `READY_FOR_EXEC`.
- Todas as tarefas têm IDs estáveis, owner, arquivos e evidência.
- Todos os ACs mapeiam para tarefas e validação.
- Implementação futura altera apenas `src/ab_cli/commands/auto_commit.py` e `tests/integration/test_auto_commit.py`, salvo descoberta bloqueante registrada antes.
