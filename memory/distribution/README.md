# Distribuição

## Responsabilidade

Mapear os artefatos Linux do `ab` e sua publicação automatizada.

## Componentes

- [PACKAGING.md](PACKAGING.md): Snap, Flatpak, atualização e release no GitHub.

## Relações

```mermaid
flowchart LR
  Tag[tag v1.0.0] --> Workflow[release.yml]
  Workflow --> Snap[Snap classic]
  Workflow --> Flatpak[Flatpak bundle]
  Workflow --> Release[GitHub Release]
```

