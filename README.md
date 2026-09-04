# ab — utilitários de desenvolvimento para Linux

`ab` reúne comandos de terminal para fluxos Git, prompts com contexto, consulta de modelos, geração de scripts e ferramentas de mídia. Os recursos de IA usam a API do OpenRouter.

## Instalação

### Script de instalação

Requer Python 3.9+; a instalação remota também requer `git`.

```bash
curl -fsSL https://raw.githubusercontent.com/allanbatista/ai-linux-dev-utilities/master/install.sh | bash
```

Opções úteis:

```bash
# Escolher o diretório de instalação
curl -fsSL https://raw.githubusercontent.com/allanbatista/ai-linux-dev-utilities/master/install.sh | bash -s -- -d ~/apps/ab

# Não pedir confirmação
curl -fsSL https://raw.githubusercontent.com/allanbatista/ai-linux-dev-utilities/master/install.sh | bash -s -- -y
```

Para instalar a partir de um clone local:

```bash
git clone https://github.com/allanbatista/ai-linux-dev-utilities.git
cd ai-linux-dev-utilities
./install.sh
```

O script cria o ambiente virtual, instala dependências, oferece o link em `/usr/local/bin/ab` e a conclusão para Bash.

### Snap e Flatpak

Os pacotes são anexos das [GitHub Releases](https://github.com/allanbatista/ai-linux-dev-utilities/releases), não do Snap Store nem do Flathub. Baixe o asset da versão desejada antes de instalar.

```bash
# Snap: ab_<versão>_amd64.snap
sudo snap install --classic --dangerous ./ab_*.snap
ab help

# Flatpak: io.github.allanbatista.ab_<versão>_x86_64.flatpak
flatpak remote-add --user --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak install --user --bundle ./io.github.allanbatista.ab_*.flatpak
flatpak run io.github.allanbatista.ab help
```

Snap requer `snapd`; Flatpak requer `flatpak`. Os dois pacotes são para x86_64.

## Pré-requisitos e configuração

| Recurso | Necessário |
| --- | --- |
| Comandos com IA | `OPENROUTER_API_KEY` |
| Instalação por script ou fonte | Linux, Bash e Python 3.9+ |
| Fluxos Git | repositório Git; `gh` para criar PRs |
| `ab media` em instalação por fonte | `ffmpeg` e `ffprobe` |

```bash
export OPENROUTER_API_KEY="sua-chave"
ab config init
ab config init --force
ab help
```

A configuração fica em `~/.ab/config.json`; o histórico de interações fica em `~/.ab/history/`.

```bash
ab config show
ab config get models.default
ab config set global.language pt-br
ab config set models.default "openai/gpt-5-nano"
ab config set commands.media.transcription_model "openai/gpt-4o-mini-transcribe"
ab config set commands.media.diarization_model "x-ai/grok-stt-1.0"
ab config path
ab config edit
ab config list-keys
ab config clear-history -y
```

As opções de linha de comando têm precedência sobre a configuração. Os comandos de IA aceitam `--reasoning-effort` (`xhigh`, `high`, `medium`, `low`, `minimal`, `none`) e `--service-tier` (`default`, `flex`, `priority`) quando aplicável.

Modelos automáticos para `auto-commit`, `pr-description` e `rewrite-history`:

| Contexto estimado | Chave de configuração padrão |
| --- | --- |
| até 128k tokens | `models.small` |
| até 256k tokens | `models.medium` |
| acima de 256k tokens | `models.large` |

## Comandos

```text
ab <git|util|media|prompt|config|models|upgrade> [argumentos]
```

| Comando | Uso |
| --- | --- |
| `ab prompt` | Envia prompt e arquivos/diretórios ao OpenRouter |
| `ab config` | Lê e altera a configuração local |
| `ab models` | Lista e consulta modelos disponíveis no OpenRouter |
| `ab upgrade` | Atualiza uma instalação feita por script/fonte |
| `ab git` | Automação de Git com IA |
| `ab util` | Explicação, geração de scripts e senhas |
| `ab media` | Extração de áudio e transcrição |

Use `ab <categoria> help` ou `ab <categoria> <comando> --help` para a referência completa de cada comando.

### Prompt e modelos

`ab prompt` combina arquivos de texto, ignora binários e respeita `.aiignore` com sintaxe compatível com `.gitignore`.

```bash
# Pergunta simples
ab prompt -p "Explique decorators em Python"

# Contexto de arquivo ou diretório
ab prompt src/app.py -p "Revise este código"
ab prompt src/ -p "Liste riscos de segurança"

# Entrada padrão, modelo específico e saída tratada como JSON
printf 'resuma isto' | ab prompt -p -
ab prompt --model "openai/gpt-5-nano" -p "Olá"
ab prompt dados.json -p "Valide o JSON" --only-output --json
```

Opções principais: `--lang`, `--model`, `--set-default-model`, `--max-tokens`, `--max-tokens-doc`, `--max-completion-tokens`, `--unlimited`, `--specialist dev|rm`, `--only-output`, `--json`, `--relative-paths` e `--filename-only`.

```bash
ab models list --free --search llama
ab models list --context-min 128000 --sort price
ab models list --modality image --json
ab models info openai/gpt-4o
ab models info openai/gpt-4o --json
```

`ab models list` aceita `--free`, `--search`, `--context-min`, `--modality`, `--limit`, `--sort name|context|price` e `--json`.

### Git

| Comando | Finalidade |
| --- | --- |
| `ab git auto-commit` | Cria mensagem de commit a partir das alterações |
| `ab git branch-name` | Sugere ou cria nome de branch por descrição |
| `ab git changelog` | Gera notas de versão a partir dos commits |
| `ab git pr-description` | Gera título e descrição de PR |
| `ab git resolve-conflict` | Analisa conflitos de merge e sugere resolução |
| `ab git rewrite-history` | Reescreve mensagens de commits com backup |

```bash
# Adiciona tudo, confirma sem pergunta, envia e cria PR
ab git auto-commit -y -Y -p -P

# Usa somente o que já está no stage
ab git auto-commit -s -Y

# Sugere/cria uma branch
ab git branch-name "corrigir login mobile"
ab git branch-name -c --prefix fix "corrigir login mobile"

# Gera changelog para um intervalo
ab git changelog v1.0.0..v1.1.0 --categories -o CHANGELOG.md

# Gera ou cria um PR
ab git pr-description
ab git pr-description -c -b develop -y       # Draft por padrão
ab git pr-description -c --ready -y           # Pronto para revisão

# Prévia segura de resolução ou reescrita
ab git resolve-conflict --dry-run
ab git rewrite-history HEAD~5..HEAD --dry-run
```

`auto-commit -P` requer `-p` e cria PR draft por padrão; use `--ready` para criá-lo pronto para revisão. `pr-description -c` segue o mesmo padrão. Ambos exigem `gh` autenticado; se já houver PR da branch para a base, exibem a URL existente. `rewrite-history` cria uma branch de backup e deve ser usado com cuidado em commits já publicados.

Opções principais: `auto-commit` usa `-f`, `-y/-a`, `-Y`, `-s`, `-p`, `-P` e `--ready`; `branch-name` usa `-c`, `--prefix` e `-y`; `changelog` usa `--format`, `--output` e `--categories`; `pr-description` usa `--base`, `-c`, `--ready` e `-y`; `resolve-conflict` usa `-y` e `--dry-run`; `rewrite-history` usa `--smart`, `--force-all`, `--skip-merges`, `--include-merges` e `--backup-branch`.

### Utilitários

```bash
# Explica arquivo, faixa de linhas, erro ou conceito
ab util explain src/app.py:42
ab util explain --concept "dependency injection"
ab util explain --history 20 "ECONNREFUSED"

# Gera uma linha de comando, script completo ou executa o resultado
ab util gen-script "listar arquivos maiores que 100MB"
ab util gen-script --full --lang python -o backup.py "fazer backup do banco"
ab util gen-script --type cron "backup diário às 3h"
ab util gen-script --run "mostrar uso de disco"

# Gera senha segura
ab util passgenerator 20 --min-digits 4 --min-punct 2
ab util passgenerator 16 --no-punct
```

`passgenerator` também aceita `--no-letters`, `--no-digits`, `--min-letters`, `--allow-sequences` e `--allow-repeated`.

`explain` aceita `--with-files`, `--context-dir` e `--verbose`; `gen-script` aceita `--output`, `--full`, `--run`, `--type`, `--lang` e `--output-lang`.

### Mídia

```bash
# Extrai áudio de vídeo
ab media extract-audio video.mp4
ab media extract-audio video.mp4 -o audio.wav --format wav

# Transcreve áudio ou vídeo pelo OpenRouter (plain text)
ab media transcribe reuniao.mp3 -o reuniao.txt --language pt
ab media transcribe video.mp4 --chunk-seconds 300 --temperature 0
ab media transcribe audio.mp3 --model openai/gpt-4o-mini-transcribe

# Com labels de falantes (Speaker 0, Speaker 1, ... — não são nomes reais)
ab media transcribe reuniao.mp3 --diarize
ab media transcribe call.wav --diarize --model x-ai/grok-stt-1.0
```

`extract-audio` aceita `mp3`, `wav`, `m4a` e `flac`; use `-y` para sobrescrever a saída. A transcrição aceita vídeo (`mp4`, `avi`, `webm`, …) ou áudio (`mp3`, `wav`, …) via `ffmpeg`. Modelo padrão: `commands.media.transcription_model`. Com `--diarize`, o default é `commands.media.diarization_model` (`x-ai/grok-stt-1.0` no OpenRouter).

## Atualização

```bash
# Instalação por script/fonte: exige checkout limpo
ab upgrade
ab upgrade --dry-run

# Snap
snap refresh ab

# Flatpak
flatpak update io.github.allanbatista.ab
```

Em instalações Snap e Flatpak, `ab upgrade` mostra o comando do gerenciador correspondente.

## Desenvolvimento

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src/ab_cli --cov-report=term-missing
```

Não inclua `~/.ab/config.json`, chaves de API ou arquivos sensíveis em commits.

## Licença

MIT.
