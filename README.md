# Clipador

App web local que transforma VODs de streamers (Twitch, YouTube, Kick) em clips verticais 9:16
legendados para YouTube Shorts, TikTok e Instagram Reels. Tudo roda local e de graça.

> Só clipe streamers que **autorizam clipes**. O Clipador registra a autorização (data e observação)
> de cada streamer e bloqueia fontes de quem não tem autorização confirmada. Nada é publicado sem
> passar pela tela de revisão.

## Status

Passo 1 (fundação): cadastro de streamers e canais, ingest de VODs/clips das três plataformas,
fila de jobs com worker separado, diagnóstico do ambiente (`doctor`).

## Requisitos (Windows 11)

1. **Python 3.11+**: <https://www.python.org/downloads/> (marque "Add python.exe to PATH").
2. **uv**: no PowerShell, `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   (ou `python -m pip install --user uv`).
3. **ffmpeg + ffprobe**: baixe o build "essentials" em <https://www.gyan.dev/ffmpeg/builds/>,
   extraia e coloque a pasta `bin` no PATH. Confira com `ffmpeg -version`.
4. **Driver NVIDIA** atualizado (para NVENC). Sem GPU o app usa `libx264` automaticamente.
5. **Git** (opcional, para versionar): <https://git-scm.com/download/win>.
6. **Ollama** (opcional por enquanto, usado a partir do dia 3): <https://ollama.com/download>.

O `yt-dlp` é instalado pelo `uv sync` como dependência do projeto.

## Instalar e rodar

```powershell
cd C:\caminho\para\clipador
uv sync
copy .env.example .env      # opcional
uv run clipador doctor      # verifica o ambiente
```

Em dois terminais separados:

```powershell
uv run clipador web         # http://127.0.0.1:8000
uv run clipador worker      # processa a fila de jobs
```

O banco (SQLite) é criado e migrado automaticamente em `data/clipador.db` na primeira execução.

## Fluxo básico

1. **Streamers** → cadastre o streamer, marque "Autorização confirmada", preencha data e observação
   (ex.: "liberou no Discord em 10/11") e adicione os canais (Twitch, YouTube, Kick).
2. **Fontes** → cole a URL de um VOD ou clip. A plataforma é detectada sozinha. Escolha o streamer
   e, se quiser, o trecho a analisar.
3. **Jobs** → acompanhe o download. Se o worker cair no meio, ao subir de novo o job volta para a
   fila e continua de onde parou.
4. **Sistema** → resultado do diagnóstico.

## Quando o download falha

- **"extrator desatualizado" / erros estranhos**: atualize o yt-dlp:
  `uv lock --upgrade-package yt-dlp` e depois `uv sync`.
- **YouTube pedindo login ("confirme que não é um robô") ou Kick bloqueando (anti-bot)**: configure
  cookies do navegador no `.env`:
  `CLIPADOR_COOKIES_FROM_BROWSER=firefox` (o Chrome no Windows costuma travar o arquivo de cookies
  enquanto está aberto; feche-o ou use Firefox) ou exporte um `cookies.txt` e use
  `CLIPADOR_COOKIES_FILE=C:\caminho\cookies.txt`.
- **VOD só para inscritos/membros**: precisa de cookies de uma conta com acesso.

## Configuração

Variáveis no `.env` (veja `.env.example`) ou num `config.toml` na pasta do projeto, com os mesmos
nomes em minúsculas e sem o prefixo `CLIPADOR_`:

```toml
data_dir = "D:/clipador-data"
cookies_from_browser = "firefox"
```

## Desenvolvimento

```powershell
uv run pytest
uv run ruff check .
uv run ruff format .
uv run alembic revision --autogenerate -m "descricao"   # nova migração após mudar models.py
```

Estrutura:

- `src/clipador/platforms/` — um adaptador por plataforma (Twitch, YouTube, Kick) atrás de
  `PlatformAdapter`; chat via `ChatProvider` (dia 2).
- `src/clipador/pipeline/` — estágios plugáveis (`ingest` hoje; `transcribe`, `detect`, `render` depois).
- `src/clipador/worker/` — fila persistida no SQLite e o loop do worker (lock único de GPU).
- `src/clipador/services/` — ffmpeg/ffprobe, yt-dlp, doctor.
- `src/clipador/web/` — FastAPI + Jinja2 + HTMX, interface em PT-BR.
