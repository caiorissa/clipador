# Clipador

App web local que transforma VODs de streamers (Twitch, YouTube, Kick) em clips verticais 9:16
legendados para YouTube Shorts, TikTok e Instagram Reels. Tudo roda local e de graça.

> Só clipe streamers que **autorizam clipes**. O Clipador registra a autorização (data e observação)
> de cada streamer e bloqueia fontes de quem não tem autorização confirmada. Nada é publicado sem
> passar pela tela de revisão.

## Status

- Passo 1 (fundação): cadastro de streamers e canais, ingest de VODs/clips das três plataformas,
  fila de jobs com worker separado, diagnóstico do ambiente (`doctor`).
- Passo 2 (detecção): análise em 4 etapas (extrair áudio → transcrever na GPU → chat → detectar),
  página da fonte com linha do tempo, player dos candidatos, feedback e reanálise instantânea,
  e calibração com `clipador dump` / `clipador eval`.

## Detecção de momentos (Passo 2)

Na página da fonte (`/fontes/<id>`), **Analisar** roda a cadeia completa. Cada etapa fica salva em
`data/work/<id>/` e só é refeita se os parâmetros dela mudarem; **Reanalisar** (painel "Ajustes
da detecção") roda só a detecção e leva segundos.

| Etapa | O que faz | Arquivo |
|---|---|---|
| Extrair áudio | WAV 16 kHz mono do trecho escolhido | `audio.wav` |
| Transcrever | faster-whisper na GPU, em blocos de 10 min (retoma do último bloco se cair) | `transcript.json.gz` |
| Chat | Twitch via TwitchDownloaderCLI, YouTube via replay de chat do yt-dlp; Kick indisponível | `chat.jsonl.gz` |
| Detectar | features por segundo + curva de interesse + picos → candidatos | `features.npz` + banco |

Sem chat a etapa fica como "pulado" e a nota usa só áudio e fala (os pesos se renormalizam).
Para o chat da Twitch, baixe o **TwitchDownloaderCLI** em
<https://github.com/lay295/TwitchDownloader/releases> e aponte `CLIPADOR_TWITCH_DOWNLOADER_PATH`.

**Palavras de empolgação:** edite `data/hype_words.toml` (criado na primeira análise). Mudar o
arquivo faz a próxima reanálise recalcular as features (rápido; não retranscreve).

**Modelo do Whisper** (`.env`), medido numa RTX 5060 8 GB com 10 min de fala:

| `CLIPADOR_WHISPER_MODEL` | `CLIPADOR_WHISPER_COMPUTE_TYPE` | velocidade | por hora de VOD | pico de VRAM* |
|---|---|---|---|---|
| `medium` (padrão) | `float16` | 21x | ~2,8 min | 3,2 GB |
| `large-v3-turbo` | `int8_float16` | 39x | ~1,5 min | 2,1 GB |
| `large-v3` | `float16` | 16x | ~3,8 min | 5,0 GB |

\* GPU inteira, incluindo ~1 GB já usado pelo Windows. Se a VRAM apertar (jogo/OBS abertos), use
`int8_float16`. Idioma: `CLIPADOR_LANGUAGE=pt` (padrão) ou `auto`. Meça no seu PC com:

```powershell
uv run clipador bench-transcribe data\vods\1\video.mp4 --seconds 600
```

**Calibração:** marque à mão os momentos bons num `golden.txt` (um `HH:MM:SS` por linha) e rode:

```powershell
uv run clipador dump 1 --top 20          # top 20 no terminal + candidates.csv
uv run clipador eval 1 --golden golden.txt   # recall@5/10/20, precisão@N e momentos perdidos
```

Os botões Bom/Ruim da página entram na precisão@N.

## IA local (Passo 3, opcional)

Com o [Ollama](https://ollama.com/download) rodando, o estágio **IA (Ollama)** (último da cadeia
Analisar, ou o botão **Enriquecer com IA**) avalia os melhores K candidatos (padrão 3 × 5 clips)
e escreve o pacote de postagem:

- notas da IA (humor, surpresa, habilidade, emoção, faz sentido sozinho, força do início, geral),
  o motivo, "precisa de contexto" e **alertas** de monetização (com o trecho que disparou);
- 3 títulos, texto de abertura, descrição e hashtags; o código monta os rascunhos de **YouTube
  Shorts, TikTok e Instagram Reels** e sempre acrescenta o **crédito do streamer** (e o link).
- **Nota final** = peso × detecção + peso × IA (padrão 50/50), com penalidades leves para clips
  que precisam de contexto. Mudar os pesos reordena na hora, sem chamar a IA.

Sem Ollama tudo continua funcionando: a etapa fica "pulado", vale a nota de detecção e cada
candidato ganha um rascunho simples (primeira frase do trecho), editável.

```powershell
# 1) instale o Ollama: https://ollama.com/download  (ele fica rodando na bandeja)
ollama pull qwen2.5:7b                 # ~4,7 GB; alternativas: llama3.1:8b, qwen3:8b
uv run clipador doctor                 # confere Ollama, modelo e faz uma geração de teste
uv run clipador llm-bench --n 10 --models qwen2.5:7b,llama3.1:8b   # qual modelo vale no seu PC
uv run clipador eval 1 --golden golden.txt --with-llm              # a IA melhorou ou piorou?
```

Configuração (`.env`): `CLIPADOR_LLM_MODEL`, `CLIPADOR_TARGET_CLIPS`, `CLIPADOR_LLM_NUM_CTX`,
`CLIPADOR_RANK_WEIGHT_DETECT` / `CLIPADOR_RANK_WEIGHT_LLM`, `CLIPADOR_DEFAULT_GAME`. O prompt fica
em `prompts/enrich_v1.md` (veja `prompts/README.md`); mudar o prompt, o modelo ou pedir "Regerar"
com uma instrução invalida o cache só do que mudou. Rascunhos que você editou nunca são
sobrescritos sem você marcar "sobrescrever o que eu editei".

**GPU:** a transcrição e a IA nunca rodam juntas (lock de GPU do worker). O Whisper libera a VRAM
ao terminar (medido: ~2,1 GB → volta a ~170 MB acima do normal) e o modelo do Ollama é
descarregado no fim do estágio (`keep_alive=0`); os dois valores aparecem no log do job.

> ⚠️ A IA não detecta **música com direitos autorais** de fundo: confira antes de postar.

## Cortes prontos (Passo 4)

Na página da fonte, em cada candidato (seção **Corte vertical 9:16**):

1. Escolha o **layout** e o **estilo de legenda** (os padrões vêm do streamer).
2. **👁 Prévia rápida** (540×960): mesma geometria e legenda do final, pronta em segundos.
3. **🎬 Renderizar final** (1080×1920, H.264 + AAC, faststart): MP4 em `data/clips/<fonte>/`
   mais um `.txt` com título/descrição/hashtags/crédito de cada plataforma. Ou use
   **Renderizar aprovados** para todos de uma vez.

- **Layouts:** *Fundo borrado*; *Facecam embaixo* (jogo em cima, câmera embaixo); *Jogo cheio*
  (recorte 9:16 do jogo, com **zoom** e **foco horizontal** 0–1). Sem facecam definida, cai para
  fundo borrado e avisa na tela.
- **Facecam:** na página do streamer, arraste um retângulo sobre um quadro de uma fonte dele
  (precisa de pelo menos uma fonte baixada). Fica salvo em coordenadas relativas (0–1): vale
  para 720p, 1080p, 1440p, 21:9…
- **Legendas palavra por palavra** (ASS) com a fonte embutida **Anton** (licença OFL, em
  `assets/fonts/`), 2–4 palavras por vez, no máximo 2 linhas, nunca menos de 0,25 s na tela, dentro
  da **zona segura** (acima dos ~20% de baixo e abaixo dos ~10% de cima, que a interface das
  plataformas cobre). Estilos editáveis em `config/caption_styles.toml` (`classico`,
  `destaque_amarelo`, `pop`, `caixa`, `neon`). Correções de palavras ficam em
  `caption_overrides` e mantêm os tempos (o editor completo vem no Passo 5).
- **Texto de abertura** (2 s, com fade) e **crédito "via @streamer"** discreto durante todo o clip;
  liga/desliga por streamer e por corte. **Zoom rápido no pico** opcional.
- **Áudio** em **−14 LUFS** (ganho medido + limitador de pico, ±1,5 LU), fades de 80 ms, estéreo;
  vídeo em **taxa constante** (VFR da Twitch vira CFR, até 60 fps).
- Cada render é **validado com ffprobe** (resolução, codecs, áudio, duração ±0,15 s, faststart) e
  só ganha o nome final depois disso; cancelar ou falhar não deixa arquivo pela metade.
  Renderizar de novo com os mesmos ajustes reaproveita o arquivo.
- Por padrão, o render espera o mesmo lock de GPU do Whisper/Ollama
  (`CLIPADOR_RENDER_PARALLEL_WITH_GPU_JOBS=false`); um render por vez.
- **Vídeo longo 16:9:** junta os aprovados em ordem cronológica, com os **capítulos** para o YouTube.

**Velocidade medida** (RTX 5060, clip de 40 s 1080p60, inclui medir o áudio):

| encoder | prévia | final | tamanho do final |
|---|---|---|---|
| NVENC (`h264_nvenc`) | 7,7x tempo real (5 s) | 4,7x (8,5 s) | ~55 MB |
| libx264 | 7,9x (5 s) | 4,0x (10 s) | ~12 MB |

Meça no seu PC (com um candidato real ou um vídeo de teste):

```powershell
uv run clipador render-bench            # vídeo de teste 1080p60 de 40 s
uv run clipador render-bench 12         # candidato #12
```
## Requisitos (Windows 11)

1. **Python**: o projeto está fixado em **3.12** (`.python-version`), por compatibilidade com
   faster-whisper/ctranslate2. Não precisa instalar à mão: o `uv sync` baixa o 3.12 sozinho.
2. **uv**: no PowerShell, `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   (ou `python -m pip install --user uv`).
3. **ffmpeg + ffprobe**: baixe o build "essentials" em <https://www.gyan.dev/ffmpeg/builds/>,
   extraia e coloque a pasta `bin` no PATH. Confira com `ffmpeg -version`.
4. **Driver NVIDIA** atualizado (para NVENC). Sem GPU o app usa `libx264` automaticamente.
5. **Deno** (recomendado para o YouTube): o yt-dlp precisa de um runtime JavaScript para achar
   todos os formatos do YouTube. PowerShell: `irm https://deno.land/install.ps1 | iex`.
6. **Git** (opcional, para versionar): <https://git-scm.com/download/win>.
7. **Ollama** (opcional por enquanto, usado a partir do dia 3): <https://ollama.com/download>.

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
Rode só um worker por pasta de dados (um segundo worker se recusa a subir). Se o worker for
fechado no meio de um job, os processos do yt-dlp/ffmpeg morrem junto e, ao subir de novo, o job
volta para a fila e continua o download de onde parou.

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
