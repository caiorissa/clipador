# Restaurar o Clipador no Mac

Este guia leva o projeto do PC (Windows) para o MacBook sem perder nada.

## O que existe e onde está

| O quê | Onde fica a cópia |
|---|---|
| Código e histórico (todos os commits) | GitHub privado `caiorissa/clipador` **e** `clipador.bundle` dentro do backup |
| Banco (streamers, fontes, candidatos, decisões, renders…) | `data/clipador.db` no backup |
| VODs baixados, cortes renderizados, transcrições, features | `data/vods`, `data/clips`, `data/work` no backup |
| Modelos do Whisper já baixados | `data/models` no backup (opcional: baixam de novo sozinhos) |
| Palavras de empolgação editadas | `data/hype_words.toml` no backup |
| Configuração local | `.env` no backup |
| Arquivos em andamento que ainda não estavam no git | pasta `wip/` no backup |
| Memória e histórico do Claude Code deste projeto | pasta `claude/` no backup |

Os caminhos no banco são relativos à pasta `data/`, então tudo continua funcionando em outra
máquina e outro sistema.

## 1. Programas no Mac

Instale o [Homebrew](https://brew.sh) e depois, no Terminal:

```bash
brew install uv ffmpeg git gh
```

O `ffmpeg` do Homebrew já vem com libass (legendas) e os filtros que o Clipador usa.
Opcionais: `brew install deno` (formatos do YouTube) e o app do [Ollama](https://ollama.com/download)
(IA local; no Mac ele usa a GPU da Apple pelo Metal).

## 2. Código

```bash
gh auth login                      # entra na conta caiorissa pelo navegador
gh repo clone caiorissa/clipador ~/Clipador
cd ~/Clipador
```

Sem internet ou sem GitHub, dá para clonar do arquivo do backup:
`git clone /caminho/do/backup/clipador.bundle ~/Clipador`.

## 3. Dados e configuração

Extraia o backup (dê dois cliques no `.tar` ou rode `tar -xf Clipador-backup-*.tar`) e copie:

```bash
BK=~/Downloads/Clipador-backup          # pasta onde você extraiu o backup
cp -R "$BK/data" ~/Clipador/
cp "$BK/.env" ~/Clipador/.env
cp "$BK"/wip/src/clipador/platforms/live.py ~/Clipador/src/clipador/platforms/ 2>/dev/null
cp "$BK"/wip/src/clipador/services/live.py ~/Clipador/src/clipador/services/ 2>/dev/null
```

Ajuste o `.env`: a linha `CLIPADOR_TWITCH_DOWNLOADER_PATH` aponta para um `.exe` do Windows.
Baixe a versão **macOS** do TwitchDownloaderCLI em
<https://github.com/lay295/TwitchDownloader/releases>, extraia e aponte para ela, por exemplo:

```bash
chmod +x ~/tools/TwitchDownloaderCLI
xattr -d com.apple.quarantine ~/tools/TwitchDownloaderCLI   # libera o binário baixado
# no .env:
# CLIPADOR_TWITCH_DOWNLOADER_PATH=/Users/<voce>/tools/TwitchDownloaderCLI
```

## 4. Ambiente Python e conferência

```bash
cd ~/Clipador
uv sync                      # instala o Python 3.12 e as dependências
uv run clipador migrate      # confere o banco
uv run clipador doctor       # mostra o que está ok no Mac
uv run pytest -q             # a suíte deve passar
```

Depois é igual ao PC: `uv run clipador web` e `uv run clipador worker` em dois terminais.

## 5. Diferenças do Mac (esperadas)

- **Sem GPU NVIDIA:** o doctor vai avisar "CUDA indisponível" e "NVENC indisponível". Isso é
  normal. O Whisper roda na CPU (int8) e o render usa `libx264` automaticamente.
  - Meça a transcrição: `uv run clipador bench-transcribe data/vods/<id>/video.mp4 --seconds 300`.
    Se ficar lento, use `CLIPADOR_WHISPER_MODEL=small` ou `large-v3-turbo` com
    `CLIPADOR_WHISPER_COMPUTE_TYPE=int8` no `.env`.
  - Meça o render: `uv run clipador render-bench`.
- **Ollama no Mac usa a GPU da Apple** (Metal): a etapa de IA tende a ficar rápida.
- As pacotes `nvidia-*` do `pyproject.toml` só instalam fora do macOS; não precisa mexer.

## 6. Claude Code (memória e histórico)

O Claude Code guarda a memória do projeto em `~/.claude/projects/<chave>/memory`, onde a chave é o
caminho da pasta com `/` trocado por `-`. Para `~/Clipador` num usuário `caio`, a chave fica
`-Users-caio-Clipador`. Abra o Claude Code uma vez na pasta do projeto (para ele criar a pasta) e
depois copie:

```bash
KEY="$(pwd | sed 's#[/.]#-#g')"                 # rode dentro de ~/Clipador
mkdir -p ~/.claude/projects/"$KEY"/memory
cp -R "$BK"/claude/memory/* ~/.claude/projects/"$KEY"/memory/
```

Os históricos das conversas (`claude/*.jsonl`) ficam no backup para consulta; eles foram gravados
com caminhos do Windows.
