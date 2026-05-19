# Status

READY_FOR_PLAN

# Goal

Corrigir o fluxo de `ab git auto-commit` para que usuários possam executar `-y -Y -p -P` a partir de `master` ou `main` sem interação manual. O comando deve preservar a proteção da branch principal, criar automaticamente a branch sugerida, fazer commit, push e abrir PR.

# Users & Journeys

- Usuário CLI em branch protegida: executa `ab git auto-commit -y -Y -p -P` em `master` ou `main`; o comando cria a branch sugerida sem prompt, comita as mudanças, envia a branch remota e cria o PR contra a branch original.
- Usuário CLI em branch não protegida: executa `ab git auto-commit -y -Y -p -P`; o comportamento existente permanece, com commit, push e PR na branch atual.
- Falha principal: se a criação de branch, commit, push ou PR falhar, o comando deve encerrar com erro observável e não indicar sucesso da etapa não concluída.

# Non-Functional Requirements

- Não expor secrets, tokens ou conteúdo sensível em logs.
- Manter mensagens de erro claras para uso em terminal e testes.
- Não exigir entrada interativa quando `-y` e `-Y` forem usados juntos.
- Preservar compatibilidade dos fluxos existentes de `auto-commit`, inclusive `-f`, `-s`, `-p` sem `-P`, e `-P` exigindo `-p`.

# Acceptance Criteria

- AC-1: Dado que o usuário está em `master` com mudanças commitáveis, ao executar `ab git auto-commit -y -Y -p -P`, o comando cria automaticamente a branch sugerida antes do commit, sem prompt interativo.
- AC-2: Dado que o usuário está em `main` com mudanças commitáveis, ao executar `ab git auto-commit -y -Y -p -P`, o comando cria automaticamente a branch sugerida antes do commit, sem prompt interativo.
- AC-3: Após a criação automática da branch em `master` ou `main`, o commit é criado na branch sugerida, não na branch protegida original.
- AC-4: Após o commit automático em branch sugerida, o comando faz push dessa branch e cria PR contra a branch protegida original.
- AC-5: Quando `ab git auto-commit -y -Y -p -P` é executado em uma branch não protegida, o comando continua usando a branch atual para commit, push e PR.
- AC-6: Quando `-f` é usado em branch protegida com `-y -Y -p -P`, o comando não cria branch automaticamente e mantém a regra existente de não criar PR a partir de branch protegida.
- AC-7: Quando `-P` é usado sem `-p`, o comando continua falhando com erro observável antes de qualquer commit, push ou PR.
- AC-8: Testes automatizados cobrem os fluxos de branch protegida `master`, branch protegida `main`, branch não protegida, `-f`, e `-P` sem `-p`.

# Scope

Inclui:

- Comportamento de produto do comando `ab git auto-commit` com `-y -Y -p -P`.
- Branches protegidas `master` e `main`.
- Criação automática da branch sugerida apenas quando o usuário já pediu modo não interativo e fluxo completo de PR.
- Evidência por testes automatizados e saída/efeitos observáveis do CLI.

Fora do escopo:

- Mudanças em outros comandos `ab git`.
- Novas opções de CLI.
- Alterações no formato de mensagens geradas por LLM.
- Mudanças em configuração de modelos, autenticação, instalação ou completions.

# Boundaries

- Não modificar arquivos de implementação nesta etapa de especificação.
- Testes são obrigatórios para qualquer mudança futura de código.
- Não commitar ou logar secrets.
- O fluxo deve respeitar a proteção de `master` e `main`; PR criado a partir dessas branches diretamente continua inválido.
- Não introduzir decisões de arquitetura, stack, endpoints ou schema neste documento.

# Open Questions

Nenhuma pergunta bloqueante para intenção de produto.

# Definition of Done

- O implementador consegue identificar todos os resultados visíveis sem inspecionar código para decidir produto.
- Todos os critérios de aceite têm evidência automatizada ou sinal observável do CLI.
- Nenhum arquivo de implementação foi alterado por esta especificação.
