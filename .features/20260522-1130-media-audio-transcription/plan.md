# Status

READY_FOR_EXEC

# Approach

Criar `ab media` como nova categoria raiz, com wrapper Bash e módulo Python único `ab_cli.commands.media`. Usar `ffprobe` para validar stream de áudio, `ffmpeg` para extrair/segmentar, e OpenRouter STT com payload JSON base64 conforme `/audio/transcriptions`.

## Interfaces / Contracts

- `ab media extract-audio <input> [-o output] [--format mp3|wav|m4a|flac] [-y|--yes]`
- `ab media transcribe <input> [-o output.txt] [--model MODEL] [--language LANG] [--chunk-seconds N] [--temperature FLOAT]`
- Config:
  - `commands.media.transcription_model`, default `openai/gpt-4o-mini-transcribe`
  - `commands.media.chunk_seconds`, default `300`
  - `commands.media.temperature`, default `0`
- OpenRouter STT helper: `transcribe_audio_openrouter(audio_path, audio_format, model, timeout_s, language=None, temperature=None, api_key_env=..., api_base=...)`.
- Sem mudança em APIs de comandos existentes.

## Technical Inventory

| slugs | queries | components | output types | filters/url state | retailer/industry compatibility |
| --- | --- | --- | --- | --- | --- |
| `media.extract-audio` | none | `bin/ab-media`, `ab_cli.commands.media` | audio file path | CLI args only: `--format`, `-o`, `-y` | Not applicable; local CLI |
| `media.transcribe` | OpenRouter `POST /audio/transcriptions` | `bin/ab-media`, `ab_cli.commands.media`, `ab_cli.utils.api` | stdout transcript or text file | CLI args only: `--model`, `--language`, `--chunk-seconds`, `--temperature`, `-o` | Not applicable; local CLI |

- Entrypoints: `bin/ab`, novo `bin/ab-media`, `completions/ab.bash-completion`.
- Runtime: `src/ab_cli/commands/media.py`, `src/ab_cli/utils/api.py`, `src/ab_cli/core/config.py`.
- Docs/tests: `README.md`, `tests/unit/test_media.py`, `tests/integration/test_media.py`, `tests/unit/test_config.py`.
- Endpoint oficial: `${api_base}/audio/transcriptions`, JSON com `input_audio.data`, `input_audio.format`, `model`, `language`, `temperature`.

# Phases / Task Breakdown

## F1 - Feature Docs

- `F1.S1.T1` Owner: executor. Criar `spec.md`, `plan.md`, `progress.md`. Done when: docs existem e registram escopo, tarefas e evidência.

Validation Gate F1: `rtk node /home/allanbatista/.codex/skills/feature-workflow/scripts/audit-feature-docs.mjs .features/20260522-1130-media-audio-transcription`.

## F2 - API/config

- `F2.S1.T1` Owner: executor. Arquivos: `src/ab_cli/core/config.py`, tests. Adicionar defaults `commands.media.*`. Done when: config e keys expõem defaults.
- `F2.S1.T2` Owner: executor. Arquivos: `src/ab_cli/utils/api.py`, tests. Adicionar helper STT com payload base64 e erros claros. Done when: testes validam payload e falhas.

Validation Gate F2: `rtk .venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_media.py -v`.

## F3 - CLI media

- `F3.S1.T1` Owner: executor. Arquivos: `src/ab_cli/commands/media.py`. Implementar helpers de subprocess, validação, saída e tempdir. Done when: helpers são testáveis sem rede.
- `F3.S1.T2` Owner: executor. Arquivos: `src/ab_cli/commands/media.py`. Implementar `extract-audio`. Done when: ffmpeg é chamado com formato/overwrite corretos.
- `F3.S1.T3` Owner: executor. Arquivos: `src/ab_cli/commands/media.py`. Implementar `transcribe` com segmentação e concatenação. Done when: chunks são transcritos em ordem.

Validation Gate F3: `rtk .venv/bin/python -m pytest tests/unit/test_media.py tests/integration/test_media.py -v`.

## F4 - Wiring/docs

- `F4.S1.T1` Owner: executor. Arquivos: `bin/ab`, `bin/ab-media`, completions. Adicionar categoria e opções. Done when: help lista `media`.
- `F4.S1.T2` Owner: executor. Arquivos: `README.md`. Documentar uso mínimo. Done when: README lista comandos e exemplos.

Validation Gate F4: `rtk ./bin/ab help` e `rtk ./bin/ab media help`.

## F5 - Validation

- `F5.S1.T1` Owner: executor. Rodar testes unitários novos.
- `F5.S1.T2` Owner: executor. Rodar testes de integração novos.
- `F5.S1.T3` Owner: executor. Rodar suíte completa.
- `F5.S1.T4` Owner: e2e-validator. Revisar evidências automatizadas e help output; não requer navegador por ser CLI local.

Validation Gate F5: `rtk .venv/bin/python -m pytest tests/ -v` e auditoria final dos feature docs.

## AC Traceability

| AC | Tasks | Evidence |
| --- | --- | --- |
| AC | Tasks | Validation evidence |
| --- | --- | --- |
| AC-1 | F3.S1.T2 | `tests/integration/test_media.py::test_extract_audio_invokes_ffmpeg` validates ffmpeg command and output path. |
| AC-2 | F3.S1.T1-F3.S1.T2 | `test_resolve_output_refuses_overwrite` and `test_extract_audio_refuses_existing_output` validate overwrite refusal. |
| AC-3 | F2.S1.T2, F3.S1.T3 | `test_transcribe_audio_segments_and_writes_output` validates chunk order and concatenated text. |
| AC-4 | F3.S1.T3 | `test_transcribe_video_uses_segment_media` validates video input path flows through segmentation before STT. |
| AC-5 | F3.S1.T3 | `test_transcribe_audio_segments_and_writes_output` validates model/language/chunk/temperature/output overrides. |
| AC-6 | F2.S1.T2, F3.S1.T1 | Unit tests validate missing dependency/API key/HTTP error; integration validates API failure exit. |
| AC-7 | F4.S1.T1-F4.S1.T2 | `rtk ./bin/ab help`, `rtk ./bin/ab media help`, README/completion diff. |

# Test Strategy

- Mockar `subprocess.run`, `shutil.which` e `requests.post`.
- Usar `tmp_path` para saídas e chunks falsos.
- Não chamar OpenRouter real.
- Comandos:
  - `rtk .venv/bin/python -m pytest tests/unit/test_media.py -v`
  - `rtk .venv/bin/python -m pytest tests/integration/test_media.py -v`
  - `rtk .venv/bin/python -m pytest tests/ -v`

# Risks & Rollback

- Risco: formatos suportados por modelo podem variar. Mitigação: converter chunks para MP3 e permitir override de modelo.
- Risco: arquivos longos geram muitas chamadas. Mitigação: `--chunk-seconds`.
- Rollback: remover `bin/ab-media`, módulo `media.py`, wiring, docs e defaults config.

# Gate Final

- `rtk .venv/bin/python -m pytest tests/unit/test_media.py -v`
- `rtk .venv/bin/python -m pytest tests/integration/test_media.py -v`
- `rtk .venv/bin/python -m pytest tests/ -v`
- `e2e-validator`: revisar evidências CLI locais, sem navegador, confirmando help output e testes de AC-1 a AC-7.
