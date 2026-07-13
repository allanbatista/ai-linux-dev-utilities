# Status

READY_FOR_PLAN

# Goal

Adicionar comandos de mídia ao `ab` para extrair áudio de vídeos e transcrever áudio/vídeo em texto usando OpenRouter STT, mantendo configuração e experiência consistentes com o CLI atual.

# Users & Journeys

- Pessoa usuária com vídeo local executa `ab media extract-audio video.mp4` e recebe um arquivo de áudio reproduzível.
- Pessoa usuária com áudio local executa `ab media transcribe audio.mp3` e recebe a transcrição no terminal ou em arquivo com `-o`.
- Pessoa usuária com vídeo local executa `ab media transcribe video.mp4`; o comando prepara áudio temporário, segmenta e junta as transcrições.
- Falha principal: dependência ausente, arquivo inválido ou erro da API termina com mensagem clara e código diferente de zero.

# Non-Functional Requirements

- Não chamar API real em testes.
- Usar `ffmpeg` e `ffprobe`; falhar claramente se ausentes.
- Reusar `global.api_base`, `global.api_key_env` e `global.timeout_seconds`.
- Modelo padrão de transcrição: `openai/gpt-4o-mini-transcribe`.
- Segmentar arquivos por padrão para reduzir falhas com entradas longas.
- Compatível com Python 3.9+ e padrões existentes do projeto.

# Acceptance Criteria

- AC-1: `ab media extract-audio <video>` chama `ffmpeg` e cria áudio no formato padrão `mp3` ou formato escolhido.
- AC-2: `extract-audio` não sobrescreve arquivo existente sem `-y/--yes`.
- AC-3: `ab media transcribe <audio>` segmenta o áudio, envia cada trecho ao OpenRouter STT e concatena textos na ordem.
- AC-4: `ab media transcribe <video>` extrai áudio temporário antes da segmentação/transcrição.
- AC-5: `--model`, `--language`, `--chunk-seconds`, `--temperature` e `-o` alteram o comportamento observado.
- AC-6: ausência de `OPENROUTER_API_KEY`, `ffmpeg`, `ffprobe`, arquivo inexistente ou erro HTTP gera falha clara.
- AC-7: ajuda, README e completions listam `ab media`.

## Product Inventory

| route/page | slug/id | user-visible label | visual/output type | filters | datasets/permissions | empty/loading/error/locked behavior | persona differences |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLI `ab media extract-audio` | `media.extract-audio` | `extract-audio` | Local audio file path | `--format`, `-o`, `-y` | Local file read/write, `ffmpeg`, `ffprobe` | Error message and exit 1 on missing input/dependency/overwrite refusal | None |
| CLI `ab media transcribe` | `media.transcribe` | `transcribe` | Terminal text or transcript file | `--model`, `--language`, `--chunk-seconds`, `--temperature`, `-o` | Local file read/write, `ffmpeg`, `ffprobe`, OpenRouter API key | Error message and exit 1 on missing input/dependency/API failure | None |

# Scope

Inclui:

- Nova categoria `ab media`.
- Extração de áudio local.
- Transcrição com OpenRouter `/audio/transcriptions`.
- Segmentação automática.
- Testes unitários e de integração.
- Documentação mínima.

Fora do escopo:

- Diarização, timestamps, legendas, tradução, streaming ou UI.
- Provedor OpenAI direto.
- Processamento paralelo de chunks.

# Open Questions

Nenhuma.

# Definition of Done

- Spec, plan e progress existem em `.features/20260522-1130-media-audio-transcription/`.
- Implementação cobre os ACs.
- Testes novos e suíte completa passam no venv.
