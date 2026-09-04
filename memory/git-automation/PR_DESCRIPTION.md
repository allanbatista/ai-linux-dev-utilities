# Criação de Pull Request

## Responsabilidade

Criar um pull request draft por padrão com `gh` ou reutilizar o pull request aberto já associado à branch.

## Entidades

- Título e corpo gerados para o pull request.
- URL do pull request criado ou já existente.
- Estado inicial draft por padrão; `--ready` solicita criação pronta para revisão.

## Relações

- `pr_description.create_pr()` é consumido por `pr-description -c` e por `auto-commit -P`.
- Falhas diferentes de pull request existente continuam sendo propagadas aos dois comandos.

## Fluxo

1. Executa `gh pr create --draft` para a branch base solicitada, exceto quando `--ready` é informado.
2. Retorna a URL criada quando o comando termina com sucesso.
3. Quando o `gh` informa um pull request existente e fornece uma URL `/pull/<n>`, retorna essa URL sem sobrescrever o pull request.

## Fontes no código

- `src/ab_cli/commands/pr_description.py`
- `src/ab_cli/commands/auto_commit.py`
- `tests/integration/test_pr_description.py`
