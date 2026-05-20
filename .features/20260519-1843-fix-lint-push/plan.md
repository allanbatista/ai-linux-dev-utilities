# Status

READY_FOR_EXEC

## Approach

Corrigir apenas violações bloqueantes do workflow `Lint` em push. O executor deve primeiro reproduzir localmente os gates reais, aplicar o menor diff necessário em arquivos de teste ou scripts shell, e validar com os mesmos comandos do workflow antes da entrega. Não alterar comportamento público do CLI `ab`.

## Interfaces / Contracts

Sem mudanças planejadas em interface pública, flags, nomes de comandos, configuração persistida, APIs, schemas, autenticação, billing, contratos externos, infraestrutura ou dependências de runtime.

Contratos que devem permanecer:

- Workflow `.github/workflows/lint.yml` continua disparando em `push` para `branches: ['**']`.
- Job `lint` continua executando `flake8 tests/ --max-line-length=120 --exclude=__pycache__` como bloqueante.
- Job `lint` continua executando `flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero` como não bloqueante.
- Job `shellcheck` continua validando `./bin` e `./scripts` com `severity: warning`.

## Technical Inventory / Inventário Técnico

| Item | Valor executável |
| --- | --- |
| Workflow | `.github/workflows/lint.yml` |
| Slug/id | `lint-push-check` / workflow `Lint` |
| Queries/consultas | Não há SQL/API query; consulta operacional é GitHub Actions logs para workflow `Lint` em push |
| Componentes | `.github/workflows/lint.yml`; jobs `lint` e `shellcheck`; alvos `tests/**`, `bin/**`, `scripts/**` |
| Tipo de output/saída | Check de CI verde/vermelho e logs de lint |
| Filtros/url state | Evento `push`; branch filter `branches: ['**']`; PR filter `branches: [master]` |
| Dataset/permission gate | Repositório GitHub; permissões padrão de GitHub Actions; sem dataset de aplicação |
| Compatibilidade retailer/industry | Não aplicável; CLI/utilitário técnico sem variação por retailer/industry |
| Evento | `push` em qualquer branch; `pull_request` para `master` permanece existente |
| Job bloqueante Python | `lint` |
| Comando Python bloqueante | `flake8 tests/ --max-line-length=120 --exclude=__pycache__` |
| Comando Python relatório | `flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero` |
| Job shell | `shellcheck` |
| Ação shell | `ludeeus/action-shellcheck@master` |
| Alvos shell | `./bin`, `./scripts` |
| Severidade shell | `warning` |
| Versão CI Python | `3.12` |
| Dependências de validação local | `flake8>=6.0.0`; `shellcheck`; opcional `pytest` quando testes mudarem |
| Compatibilidade | Linux/Ubuntu GitHub Actions; Python package src-layout em `src/ab_cli` |

## Affected Files

Planejados conforme diagnóstico:

- `.github/workflows/lint.yml`: somente leitura/contrato; não editar sem nova decisão.
- `tests/**`: permitido apenas para corrigir violações de `flake8 tests/`.
- `bin/**`: permitido apenas para corrigir violações de ShellCheck.
- `scripts/**`: permitido apenas para corrigir violações de ShellCheck.
- `.features/20260519-1843-fix-lint-push/progress.md`: executor deve criar e registrar evidências.

## Phases / Task Breakdown

## F1 - Diagnóstico

### F1.S1 - Reproduzir gates bloqueantes

- `F1.S1.T1` Owner: executor. Arquivos planejados: nenhum. Dependências: nenhuma. Executar `python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__` em ambiente com dev deps. Done when: saída e exit code registrados em `progress.md`. Evidência: log local ou link do GitHub Actions.
- `F1.S1.T2` Owner: executor. Arquivos planejados: nenhum. Dependências: nenhuma. Executar `shellcheck --severity=warning bin/*` ou equivalente recursivo para `./bin`. Done when: violações atuais registradas em `progress.md`. Evidência: log local ou link do GitHub Actions.
- `F1.S1.T3` Owner: executor. Arquivos planejados: nenhum. Dependências: nenhuma. Executar `shellcheck --severity=warning scripts/*` ou equivalente recursivo para `./scripts`. Done when: violações atuais registradas em `progress.md`. Evidência: log local ou link do GitHub Actions.

Validation Gate F1:

- Comandos: os três acima.
- Evidência: lista de arquivos/regras falhando, ou confirmação de que a falha só existe no GitHub Actions com link.
- Handoff e2e-validator: não aplicável nesta fase, salvo divergência entre local e CI.

## F2 - Correção mínima

### F2.S1 - Corrigir lint bloqueante

- `F2.S1.T1` Owner: executor. Arquivos planejados: somente `tests/**` apontados por `F1.S1.T1`. Dependências: `F1.S1.T1`. Corrigir violações de `flake8 tests/` sem alterar intenção dos testes. Done when: `flake8 tests/` passa. Evidência: comando e exit code.
- `F2.S1.T2` Owner: executor. Arquivos planejados: somente `bin/**` apontados por `F1.S1.T2`. Dependências: `F1.S1.T2`. Corrigir violações ShellCheck em `./bin` sem mudar comportamento CLI. Done when: ShellCheck em `./bin` passa. Evidência: comando e exit code.
- `F2.S1.T3` Owner: executor. Arquivos planejados: somente `scripts/**` apontados por `F1.S1.T3`. Dependências: `F1.S1.T3`. Corrigir violações ShellCheck em `./scripts` sem mudar comportamento. Done when: ShellCheck em `./scripts` passa. Evidência: comando e exit code.

Validation Gate F2:

- Comandos:
  - `python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__`
  - `shellcheck --severity=warning bin/*`
  - `shellcheck --severity=warning scripts/*`
- Evidência: logs limpos no `progress.md`.
- Handoff e2e-validator: revisar se o diff não relaxou regras nem removeu etapas do workflow.

## F3 - Validação final

### F3.S1 - Confirmar contrato e regressão

- `F3.S1.T1` Owner: executor. Arquivos planejados: `.github/workflows/lint.yml` leitura. Dependências: F2. Confirmar que `src/` mantém `--exit-zero` e que `tests/`, `bin`, `scripts` seguem bloqueantes. Done when: contrato comparado com spec. Evidência: trecho ou hash do workflow no `progress.md`.
- `F3.S1.T2` Owner: executor. Arquivos planejados: `tests/**` apenas se alterados. Dependências: F2. Se qualquer teste mudar, executar pytest relevante; se nenhum teste mudar, registrar justificativa para não executar pytest completo. Done when: evidência registrada. Evidência: comando pytest ou justificativa.
- `F3.S1.T3` Owner: e2e-validator. Arquivos planejados: nenhum. Dependências: F3.S1.T1 e F3.S1.T2. Validar entrega final contra ACs e evidências locais/CI. Done when: aprovação ou bloqueios registrados no `progress.md`. Evidência: checklist do e2e-validator.

Validation Gate F3:

- Comandos mínimos:
  - `python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__`
  - `python -m flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero`
  - `shellcheck --severity=warning bin/*`
  - `shellcheck --severity=warning scripts/*`
- Condicional: `python -m pytest <arquivo/teste relevante> -v` se `tests/**` mudar.
- Evidência: logs locais e, se disponível, link do GitHub Actions `Lint` verde em push.
- Handoff e2e-validator: obrigatório antes de marcar concluído.

## Test Strategy

- Lint Python bloqueante: `python -m flake8 tests/ --max-line-length=120 --exclude=__pycache__`.
- Lint Python não bloqueante/contrato: `python -m flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero`.
- Shell lint: `shellcheck --severity=warning bin/*` e `shellcheck --severity=warning scripts/*`.
- Pytest: obrigatório somente se arquivos de teste forem alterados; usar alvo mais específico possível e registrar evidência.
- CI real: preferir confirmar o check `Lint` verde no GitHub Actions após push.

## AC Traceability / Matriz AC

| AC | Task IDs | Evidência exigida |
| --- | --- | --- |
| AC-1 | F3.S1.T3 | Evidência: link ou registro do workflow `Lint` verde em push; se indisponível, logs locais equivalentes e pendência registrada. |
| AC-2 | F1.S1.T1, F2.S1.T1, F3.S1.T3 | Evidência: `flake8 tests/ --max-line-length=120 --exclude=__pycache__` com exit code 0. |
| AC-3 | F3.S1.T1, F3.S1.T3 | Evidência: `.github/workflows/lint.yml` ainda contém `flake8 src/ ... --exit-zero`. |
| AC-4 | F1.S1.T2, F2.S1.T2, F3.S1.T3 | Evidência: ShellCheck em `./bin` com severidade `warning` concluindo com exit code 0. |
| AC-5 | F1.S1.T3, F2.S1.T3, F3.S1.T3 | Evidência: ShellCheck em `./scripts` com severidade `warning` concluindo com exit code 0. |
| AC-6 | F2.S1.T1, F2.S1.T2, F2.S1.T3, F3.S1.T1 | Evidência: diff sem mudanças em comandos/flags documentados no `README.md`; se README mudar, justificar como não funcional. |
| AC-7 | F3.S1.T2 | Evidência: pytest relevante passando quando `tests/**` mudar, ou justificativa registrada quando não mudar. |

## Risks & Rollback

- Risco: correção ShellCheck alterar semântica de script. Mitigação: diff mínimo e executar comando afetado quando seguro. Rollback: reverter apenas commit/tarefa `F2.S1.T2` ou `F2.S1.T3`.
- Risco: correção em teste mascarar cobertura. Mitigação: preservar asserts e intenção. Rollback: reverter `F2.S1.T1`.
- Risco: ambiente local sem `flake8` ou `shellcheck`. Mitigação: usar ambiente dev/CI e registrar dependência. Rollback: não aplicável.

## Out of Scope

- Alterar regras, severidade ou gatilhos do workflow.
- Remover `--exit-zero` de `src/`.
- Refatorar `src/`.
- Corrigir falhas fora do check `Lint`.
- Criar comandos, flags, integrações ou documentação funcional.

## Paralelização / Subagents

- Paralelizáveis: `F1.S1.T1`, `F1.S1.T2`, `F1.S1.T3`.
- Paralelizáveis após diagnóstico: `F2.S1.T1`, `F2.S1.T2`, `F2.S1.T3`, desde que toquem arquivos distintos.
- Owner sugerido: executor principal para Python/tests; subagent shell para `bin/**` e `scripts/**` se disponível; `e2e-validator` para `F3.S1.T3`.
- Não paralelizar edição no mesmo arquivo sem coordenação.

## Gate Final

Antes de encerrar:

- `progress.md` criado com tarefas, arquivos reais, evidências e bloqueios.
- Gates F3 executados ou bloqueios/dependências registrados.
- `e2e-validator` validou ACs e evidências.
- `git diff` contém apenas correções necessárias ao lint e docs de workflow.
- Nenhuma mudança pública de CLI, configuração ou README funcional.

## Definition of Done

- Workflow `Lint` passa em push ou evidência local equivalente está registrada com pendência explícita de CI.
- Todos os ACs têm evidência no `progress.md`.
- Diff final limitado aos arquivos necessários para corrigir lint e aos documentos `.features`.
