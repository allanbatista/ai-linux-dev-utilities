# Empacotamento Linux

## Responsabilidade

Gerar artefatos Snap e Flatpak a partir do layout existente de `bin/`, `src/` e `scripts/`.

## Entidades

- Snap `ab` em confinamento `classic`.
- Flatpak `io.github.allanbatista.ab` com runtime Python e ferramentas de Git e mídia.
- Os módulos `gh` e `ffmpeg` do Flatpak usam arquivos externos verificados por `sha256` no manifesto.
- O módulo `ab` compartilha rede somente durante o build para instalar dependências Python; o acesso de rede do aplicativo continua definido em `finish-args`.
- Release GitHub disparada por tag `v*`.

## Relações

- Os dispatchers em `bin/` aceitam `AB_PYTHON` para usar o interpretador incluído no pacote.
- `ab upgrade` delega a atualização ao Snap ou Flatpak quando executado dentro desses ambientes.
- O workflow atualiza o índice APT e constrói o Snap no runner efêmero com `--destructive-mode`, constrói o Flatpak e anexa os dois artefatos à release da tag validada.
- O `flatpak-builder` executa `pip install` do módulo `ab` com `--share=network`, necessário para resolver as dependências no sandbox de build.

## Fluxo

1. A tag em `master` valida versão, testes e lint.
2. O workflow constrói os artefatos Snap e Flatpak.
3. A GitHub Release recebe os artefatos e as notas geradas automaticamente.

## Fontes no código

- `packaging/snap/snapcraft.yaml`
- `packaging/flatpak/io.github.allanbatista.ab.yml`
- `.github/workflows/release.yml`
- `bin/ab-upgrade`
