# Status

READY_FOR_PLAN

# Goal

Evitar que `ab git auto-commit` envie conteúdo ou metadados de arquivos binários para o LLM ao gerar branch e mensagem de commit, reduzindo vazamento acidental de dados e uso desnecessário de tokens sem impedir que esses arquivos sejam commitados.

# Users & Journeys

- Pessoa desenvolvedora com mudanças textuais e binárias staged: executa `ab git auto-commit`; o comando gera uma mensagem baseada somente nas mudanças textuais e ainda permite criar o commit com todos os arquivos staged.
- Pessoa desenvolvedora com `-y`/`-a`: executa `ab git auto-commit -y -Y` com arquivos binários não staged; o comando pode stagear os arquivos normalmente, mas o contexto enviado ao LLM não inclui os binários.
- Pessoa desenvolvedora em modo staged-only: executa `ab git auto-commit -s -Y`; arquivos binários staged são ignorados no contexto de geração, enquanto arquivos textuais staged continuam usados.
- Falha principal: quando não houver nenhuma mudança textual staged utilizável para contexto, o comando encerra com aviso observável e não chama o LLM.

# Non-Functional Requirements

- Não expor conteúdo binário, nomes de arquivos binários ou resumos de status de arquivos binários no prompt enviado ao LLM.
- Não alterar a interface pública, flags, idioma configurado, fluxo de confirmação, push, PR ou proteção de branch do comando.
- Manter mensagens de terminal claras para sucesso, ausência de contexto textual e erro.
- Preservar compatibilidade com Python 3.12 e com a suíte de testes existente.
- Testes automatizados são obrigatórios para qualquer mudança futura de código.

# Acceptance Criteria

- AC-1: Dado um commit staged com um arquivo textual e um arquivo binário, ao executar `ab git auto-commit`, o prompt enviado ao LLM contém o diff e o status do arquivo textual e não contém o conteúdo, caminho ou status do arquivo binário.
- AC-2: Dado `ab git auto-commit -y -Y` com arquivos textuais e binários não staged, o comando stageia os arquivos conforme o comportamento atual, mas o prompt enviado ao LLM inclui somente mudanças textuais.
- AC-3: Dado `ab git auto-commit -s -Y` com arquivos textuais e binários staged e arquivos não staged adicionais, o prompt enviado ao LLM inclui somente mudanças textuais staged e não inclui arquivos binários nem arquivos não staged.
- AC-4: Dado que apenas arquivos binários estão staged, `ab git auto-commit` encerra com aviso observável de que não há mudanças textuais para gerar mensagem e não realiza chamada ao LLM nem cria commit.
- AC-5: Quando arquivos binários são omitidos do contexto, eles continuam staged e disponíveis para o commit se houver pelo menos uma mudança textual válida e o usuário confirmar o commit.
- AC-6: O resumo de mudanças exibido no terminal pode continuar mostrando arquivos binários locais, mas nenhum dado de arquivo binário aparece no payload/prompt da chamada ao LLM em testes automatizados.
- AC-7: Testes automatizados cobrem os fluxos misto texto+binário, somente binário, `-y -Y`, e `-s -Y`.

# Scope

Inclui:

- Comportamento de produto do comando `ab git auto-commit` ao montar contexto para geração de branch e mensagem.
- Exclusão de arquivos binários do contexto enviado ao LLM.
- Preservação do comportamento de staging, confirmação, commit, push, PR e branch protegida.
- Evidência por testes automatizados que inspecionem o prompt/payload da chamada ao LLM e os efeitos observáveis do CLI.

Fora do escopo:

- Mudanças em `ab prompt`, `ab git pr-description`, `ab git changelog` ou outros comandos.
- Novas flags, configuração persistente ou opções de usuário.
- Alterações no formato JSON esperado do LLM.
- Alterações em modelos, autenticação, instalação, completions ou README, exceto se o plano futuro justificar documentação mínima.

# Boundaries

- Esta especificação não altera código de implementação.
- Não introduzir decisões de arquitetura, stack, arquivos, endpoints, schema ou ordem de implementação.
- Não commitar secrets nem arquivos de configuração local.
- Se a implementação futura precisar decidir como classificar arquivos binários, registrar a decisão no plano, não nesta especificação.
- Qualquer mudança que impeça commit de arquivos binários deve voltar para revisão de produto.

# Open Questions

Nenhuma pergunta bloqueante para intenção de produto.

Assunção não bloqueante: "não enviados/usados no contexto" significa excluir tanto conteúdo quanto identificação/status de arquivos binários do prompt usado para gerar branch e mensagem.

# Definition of Done

- O implementador consegue identificar todos os resultados visíveis sem inspecionar código para decidir produto.
- Todos os critérios de aceite têm evidência automatizada ou sinal observável do CLI.
- Nenhum arquivo de implementação foi alterado por esta especificação.
