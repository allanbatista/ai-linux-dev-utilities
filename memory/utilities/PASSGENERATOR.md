# Gerador de senhas

## Responsabilidade

Gerar senhas a partir de `/dev/urandom` com regras de composição opcionais.

## Entidades

- `scripts/passgenerator`: script Bash que monta o conjunto de caracteres e valida a senha gerada.

## Relações

- `bin/ab-util` encaminha `ab util passgenerator` ao script.
- Os pacotes Snap e Flatpak copiam `scripts/` para a instalação.

## Fluxo

1. Lê uma quantidade finita de bytes aleatórios e filtra o conjunto escolhido.
2. Acumula caracteres até atingir o tamanho solicitado.
3. Rejeita sequências, repetições e combinações que não atendem aos mínimos configurados.

## Fontes no código

- `scripts/passgenerator`
- `bin/ab-util`
- `tests/integration/test_passgenerator.py`
