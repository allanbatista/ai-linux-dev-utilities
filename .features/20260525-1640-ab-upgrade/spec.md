# Status

READY_FOR_PLAN

# Goal

Adicionar o comando raiz `ab upgrade` para atualizar a instalação local do `ab` pelo próprio CLI, reduzindo manutenção manual e preservando segurança sobre alterações locais do usuário.

# Users & Journeys

- Pessoa usuária executa `ab upgrade`; o CLI atualiza o checkout instalado, refresca instalação/dependências em modo não interativo e termina com sucesso observável.
- Pessoa usuária executa `ab upgrade --dry-run`; o CLI mostra os comandos/ações que executaria sem alterar arquivos, dependências, links, completions ou estado git.
- Pessoa usuária executa `ab upgrade --help`; o CLI explica objetivo, comportamento normal, opções e falhas possíveis.
- Falha principal: qualquer etapa de atualização falha com mensagem clara, código diferente de zero e sem descartar alterações locais.

# Non-Functional Requirements

- O fluxo deve ser não interativo por padrão e usar opções seguras ao chamar a rotina existente de instalação/atualização.
- O comando não deve exigir chave OpenRouter nem chamar LLM/API externa.
- Alterações locais não relacionadas devem ser preservadas: o comando não pode fazer `reset`, checkout destrutivo, stage, commit, delete ou overwrite silencioso de trabalho do usuário.
- Se o estado local impedir atualização segura, o comando deve falhar de forma explícita em vez de modificar ou esconder alterações.
- Saídas devem ser concisas, com indicação da etapa em execução, sucesso, dry-run ou falha.
- Mudança futura de código exige testes conforme `AGENTS.md`.

# Acceptance Criteria

- AC-1: `ab help` lista `upgrade` como comando raiz com descrição curta de autoatualização.
- AC-2: `ab upgrade --help` retorna sucesso e documenta o comportamento normal, `--dry-run`, execução não interativa, preservação de alterações locais e retorno não-zero em falhas.
- AC-3: `ab upgrade` executa um fluxo observável de atualização git do checkout instalado seguido de refresh de instalador/dependências em modo não interativo com flags seguras; não solicita input e retorna `0` quando todas as etapas passam.
- AC-4: `ab upgrade --dry-run` imprime os comandos/ações exatos que seriam executados e retorna `0` sem alterar HEAD, índice, worktree, ambiente virtual/dependências, symlinks ou completions.
- AC-5: Se qualquer etapa do upgrade falhar, `ab upgrade` interrompe o fluxo, mostra a etapa que falhou e retorna código diferente de zero.
- AC-6: Trabalho sujo não relacionado existente é preservado antes e depois de `ab upgrade` e `ab upgrade --dry-run`; se bloquear a atualização segura, o comando falha com mensagem clara e não altera esses paths.

## Inventário de Produto

| rota/página | slug/id | label visível | tipo de saída | filtros/opções | dados/permissões | estados vazio/loading/erro/bloqueado | diferenças por persona |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLI `ab upgrade` | `root.upgrade` | `upgrade` | Logs de terminal e exit code | `--dry-run`, `-h/--help` | Checkout instalado do `ab`, permissões locais de git/instalação/dependências | Dry-run lista comandos; sucesso retorna `0`; erro retorna não-zero; bloqueio por worktree preserva alterações | Nenhuma |

# Scope

Inclui:

- Novo comando raiz `ab upgrade`.
- Ajuda raiz e ajuda específica do comando.
- Modo normal de autoatualização via atualização git e refresh de instalador/dependências.
- Modo `--dry-run`.
- Tratamento de falhas e preservação de worktree sujo.
- Testes e documentação mínima necessários no plano futuro.

Fora do escopo:

- Atualização de configurações em `~/.ab/config.json`.
- Migração de dados, telemetria, rollback automático ou gerenciador de versões.
- Mudanças em comandos `ab git`, `ab util`, `ab media`, OpenRouter ou modelos.
- Resolver trabalho sujo existente no repositório.

# Boundaries

- Esta entrega cria somente este `spec.md`; nenhum arquivo de implementação deve ser alterado nesta etapa.
- O plano futuro deve preservar estes paths sujos não relacionados observados: `README.md`, `bin/ab`, `completions/ab.bash-completion`, `src/ab_cli/core/config.py`, `src/ab_cli/utils/__init__.py`, `src/ab_cli/utils/api.py`, `tests/unit/test_config.py`, `.features/20260522-1130-media-audio-transcription/`, `bin/ab-media`, `src/ab_cli/commands/media.py`, `tests/integration/test_media.py`, `tests/unit/test_media.py`.
- Não usar operação destrutiva de git nem sobrescrever arquivos do usuário para completar o upgrade.
- Detalhes exatos de comandos, arquivos, testes e integração pertencem ao `plan.md`, não a esta especificação.

# Open Questions

Nenhuma pergunta bloqueante para a intenção de produto.

# Definition of Done

- O implementador consegue identificar todos os resultados visíveis do `ab upgrade` sem adivinhar intenção de produto.
- Todos os critérios de aceite têm sinal observável por CLI, exit code, estado de arquivos/git ou testes.
- Nenhum arquivo de implementação foi alterado por esta especificação.
