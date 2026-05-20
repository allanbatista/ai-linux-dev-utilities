# Status

READY_FOR_PLAN

# Goal

Restaurar a confiabilidade do check de GitHub Actions `Lint` em eventos de push, garantindo que contribuições possam ser enviadas sem bloqueio por violações de lint atuais e sem alteração funcional intencional no CLI `ab`.

# Users & Journeys

- Pessoa desenvolvedora: faz push de uma branch; o workflow `Lint` executa e conclui com sucesso, permitindo confiar no sinal de qualidade antes de abrir ou atualizar PR.
- Mantenedor: revisa uma branch; vê o check `Lint` verde no GitHub Actions e não precisa distinguir falha de estilo de falha funcional.
- Falha principal: se ainda houver violação de flake8 em `tests/` ou ShellCheck em `bin/` ou `scripts/`, o workflow permanece vermelho com saída observável apontando o arquivo e a regra.

# Non-Functional Requirements

- Não alterar comportamento funcional, interface CLI, mensagens de usuário ou configuração persistente, exceto se indispensável para remover uma violação de lint.
- Não expor secrets, tokens, paths privados sensíveis ou conteúdo de `~/.ab` em logs.
- Manter compatibilidade com Python 3.12 no GitHub Actions.
- Preservar legibilidade e estilo atual de Python, Bash e testes.
- A correção deve ser pequena e verificável por comandos locais equivalentes ao workflow.

# Acceptance Criteria

- AC-1: Em evento de push, o workflow `.github/workflows/lint.yml` conclui com sucesso no GitHub Actions.
- AC-2: O job `lint` conclui com sucesso, incluindo `flake8 tests/ --max-line-length=120 --exclude=__pycache__`.
- AC-3: O job `lint` continua executando `flake8 src/ --max-line-length=120 --exclude=__pycache__ --exit-zero` como relatório não bloqueante.
- AC-4: O job `shellcheck` conclui com sucesso para `./bin` com severidade `warning`.
- AC-5: O job `shellcheck` conclui com sucesso para `./scripts` com severidade `warning`.
- AC-6: Nenhum comando público documentado em `README.md` muda de nome, flags ou comportamento esperado por causa desta correção.
- AC-7: Quando houver mudança em testes, a suíte pytest relevante passa localmente; quando não houver mudança em testes, a decisão de não executar pytest completo fica registrada no plano/progresso com justificativa.

## Product Inventory

| Página/rota | Slug/id | Label | Tipo de saída | Filtros | Datasets/permissões | Estados vazio/carregando/erro/bloqueado | Diferenças por persona |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GitHub Actions da branch enviada por push | `.github/workflows/lint.yml` | `Lint` | Check de CI com jobs `lint` e `shellcheck` | Evento `push` em qualquer branch | Repositório e permissões padrão de GitHub Actions | Carregando: check em execução; erro: job vermelho com logs de lint; bloqueado: permissões indisponíveis ou Actions desabilitado; vazio: não aplicável, pois push deve disparar execução | Pessoa desenvolvedora vê status da branch; mantenedor vê status para revisão |

# Scope

Inclui:

- Diagnosticar a falha atual do workflow `Lint` em push.
- Corrigir violações bloqueantes cobertas por `flake8 tests/` e ShellCheck em `bin/` e `scripts/`.
- Preservar o comportamento existente do CLI e dos testes.
- Registrar evidência local e/ou do GitHub Actions de que o check voltou a passar.

Fora do escopo:

- Remover `--exit-zero` do lint de `src/`.
- Refatorar código sem relação com a falha de lint.
- Alterar workflows não relacionados ao check `Lint`.
- Criar novos comandos, flags, modelos, integrações ou documentação funcional.
- Corrigir falhas de testes ou build que não sejam necessárias para o check `Lint` em push.

# Boundaries

- Esta especificação não altera código de implementação.
- Testes são obrigatórios para qualquer mudança futura de código que altere comportamento.
- Não commitar secrets nem arquivos de configuração local.
- Não mascarar a falha removendo etapas de lint, reduzindo severidade ou relaxando regras sem decisão explícita do mantenedor.
- Qualquer mudança que altere comportamento de usuário deve voltar para revisão de produto antes de execução.

# Open Questions

Nenhuma pergunta bloqueante para intenção de produto.

Assunção não bloqueante: "lint push check" se refere ao workflow `.github/workflows/lint.yml` disparado por `push`.

# Definition of Done

- O workflow `Lint` em push passa.
- As evidências locais equivalentes ao workflow estão registradas no `progress.md` da execução futura.
- O diff final contém apenas mudanças necessárias para corrigir o check de lint.
