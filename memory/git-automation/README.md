# Automação Git

## Responsabilidade

Documentar os fluxos que criam commits e pull requests pelo comando `ab`.

## Componentes

- [PR_DESCRIPTION.md](PR_DESCRIPTION.md): geração e reutilização de pull requests.

## Relações

```mermaid
flowchart LR
  PR[pr_description.create_pr] --> GH[gh pr create]
  Auto[auto-commit -P] --> PR
  Description[pr-description -c] --> PR
```

