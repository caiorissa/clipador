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
- Passo 5A (revisão): tela `/revisao` com fila global, atalhos de teclado, prévia ao vivo,
  ajuste do corte na forma de onda e histórico com desfazer.
- Passo 5B: editor de legenda palavra a palavra, glossário por streamer, verificações antes de
  publicar ("pronto para publicar") e calibração dos pesos com `clipador calibrate`.
- Passo 5C: detecção automática da facecam (`clipador facecam`, "Detectar de novo").
- Passo 5E: lives ao vivo (gravação em segmentos, análise incremental, clipagem automática com
  prévia, aviso no navegador) e prévias automáticas dos melhores candidatos.

## Detecção de momentos (Passo 2)

Na página da fonte (`/fontes/<id>`), **Analisar** roda a cadeia completa. Cada etapa fica salva em
`data/work/<id>/` e só é refeita se os parâmetros dela mudarem; **Reanalisar** (painel "Ajustes
da detecção") roda só a detecção e leva segundos.

| Etapa | O que faz | Arquivo |
|---|---|---|
| Extrair áudio | WAV 16 kHz mono do trecho escolhido | `audio.wav` |
| Transcrever | faster-whisper na GPU, em blocos de 10 min (retoma do último bloco se cair) | `transcript.json.gz` |
| Chat | Twitch via TwitchDownloaderCLI, YouTube via replay de chat do yt-dlp; Kick indisponível | `chat.jsonl.gz` |
| Detectar | features por segundo (áudio, fala, chat, reação) + curva de interesse + picos → candidatos | `features.npz` + banco |

Sem chat a etapa fica como "pulado" e a nota usa só áudio e fala (os pesos se renormalizam).
Para o chat da Twitch, baixe o **TwitchDownloaderCLI** em
<https://github.com/lay295/TwitchDownloader/releases> e aponte `CLIPADOR_TWITCH_DOWNLOADER_PATH`.

**Reação e música (eventos de áudio):** o modelo YAMNet (Google, AudioSet; ~16 MB, baixado uma
vez para `data/models`, conferido por SHA-256) roda na CPU (~5 s por hora de áudio) e reconhece
**risadas, gritos e comemoração**, que viram o sinal **Reação** da nota (o volume sozinho não
distingue um grito de uma explosão do jogo). Ele também mede a **música de fundo**: candidatos
com música em boa parte do trecho ganham o alerta "🎵 confira os direitos autorais". Desligar:
`CLIPADOR_AUDIO_EVENTS=false`. Sem internet na primeira vez, a análise segue sem esse sinal.

**Palavras de empolgação:** edite `data/hype_words.toml` (criado na primeira análise). Mudar o
arquivo faz a próxima reanálise recalcular as features (rápido; não retranscreve).

**Modelo do Whisper** (`.env`; o padrão `auto` usa `medium` no PC e `large-v3-turbo` no Mac),
medido numa RTX 5060 8 GB com 10 min de fala:

| `CLIPADOR_WHISPER_MODEL` | `CLIPADOR_WHISPER_COMPUTE_TYPE` | velocidade | por hora de VOD | pico de VRAM* |
|---|---|---|---|---|
| `medium` (padrão no PC) | `float16` | 21x | ~2,8 min | 3,2 GB |
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

> ⚠️ A IA não detecta **música com direitos autorais**; o detector de eventos de áudio avisa
> quando há música de fundo no trecho, mas não sabe se ela é liberada: confira antes de postar.

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
- **Facecam:** detectada sozinha (veja "Facecam automática (Passo 5C)"); na página do streamer
  dá para corrigir arrastando um retângulo sobre um quadro de uma fonte dele. Fica salvo em
  coordenadas relativas (0–1): vale para 720p, 1080p, 1440p, 21:9…
- **Legendas palavra por palavra** (ASS) com a fonte embutida **Anton** (licença OFL, em
  `assets/fonts/`), 2–4 palavras por vez, no máximo 2 linhas, nunca menos de 0,25 s na tela, dentro
  da **zona segura** (acima dos ~20% de baixo e abaixo dos ~10% de cima, que a interface das
  plataformas cobre). Estilos editáveis em `config/caption_styles.toml` (`classico`,
  `destaque_amarelo`, `pop`, `caixa`, `neon`). Correções de palavras ficam em
  `caption_overrides` e mantêm os tempos (o editor completo vem no Passo 5).
- **Texto de abertura** (2 s, com fade) e **crédito "via @streamer"** discreto durante todo o clip;
  liga/desliga por streamer e por corte. **Zoom rápido no pico** opcional.
- **Cortar silêncios (jump cuts)**, ligado por padrão: tira tempo morto de dentro do clip, só
  onde não há fala **e** o áudio está bem mais baixo que a voz do streamer, e nunca perto do
  pico (ação do jogo sem fala não é cortada). Legenda e zoom acompanham. Desliga por corte
  ("cortar silêncios") ou em geral com `CLIPADOR_RENDER_CUT_SILENCES=false`.
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

## Revisão (Passo 5A)

A tela **`/revisao`** (menu "Revisão") mostra **um candidato por vez**, de todas as fontes (só a
análise mais recente de cada fonte), para decidir quase só com o teclado:

- **Fila com filtros:** fonte, streamer, plataforma, nota final mínima e status (padrão:
  pendentes; os **pulados** voltam no fim da fila), ordenada por posição, nota ou horário.
  Mostra o progresso ("12 de 30"), **← Anterior**, **Desfazer** e o **histórico** das decisões.
- **Resumo:** nota final, de detecção e da IA, barras do detalhamento, motivo da IA, alertas
  (com o trecho) e o título sugerido.
- **Prévia ao vivo (aproximada):** um palco 9:16 desenhado no navegador com o vídeo original no
  layout escolhido (borrado, facecam embaixo, jogo cheio; usa a facecam, o zoom e o foco do
  corte) e as legendas sincronizadas, no estilo de `config/caption_styles.toml`. Mudar início/fim
  ou o layout atualiza na hora, sem render. O que é exportado é a **prévia fiel** (render real,
  tecla `P`): ela aparece ao lado quando fica pronta.
- **Ajuste do corte:** linha do tempo com a forma de onda do áudio (±60 s), as palavras
  transcritas e duas alças arrastáveis que **encaixam nas bordas das palavras** e nas pausas
  (desligue em "encaixar nas palavras" ou segure `Alt` ao arrastar). Há também campos numéricos
  e botões **+5 s de contexto**. Sobrepor outro corte já aprovado da mesma fonte só gera aviso.
  Os limites vêm de `CLIPADOR_CLIP_MIN_SECONDS` / `CLIPADOR_CLIP_MAX_SECONDS` (3 s / 180 s).
- **Decisões:** aprovar grava o feedback "bom"; rejeitar pede o motivo (sem graça, sem contexto,
  áudio ruim, risco de música com direitos, repetido, conteúdo sensível, outro) e grava "ruim";
  pular não grava veredito. Tudo vai para um histórico persistido: **desfazer** (`Z`) volta a
  última decisão de qualquer candidato, quantas vezes quiser. Os botões Aprovar/Rejeitar da
  página da fonte passam pelo mesmo caminho.
- **Render automático:** aprovar já enfileira o render final com os padrões do streamer
  (`CLIPADOR_RENDER_ON_APPROVE=true`), sem duplicar se já existe um igual. Ao abrir um candidato,
  a prévia fiel dos **próximos 3** pendentes entra na fila com prioridade baixa
  (`CLIPADOR_REVIEW_PREFETCH`); renders pedidos à mão passam na frente.

| tecla | ação |
|---|---|
| `A` | aprovar |
| `R` e depois `1`…`7` | rejeitar com motivo (`Esc` cancela) |
| `S` | pular |
| `Z` | desfazer (repita para voltar mais) |
| `Backspace` | voltar ao candidato anterior |
| `Espaço` | play / pause |
| `←` / `→` | −/+ 2 s no player |
| `[` / `]` | início −/+ 0,5 s (`Shift` = 2 s) |
| `,` / `.` | fim −/+ 0,5 s (`Shift` = 2 s) |
| `L` | alternar o layout |
| `P` | gerar a prévia fiel |
| `E` | ir para o editor de legenda (Passo 5B) |
| `?` | ajuda dos atalhos |

Os atalhos não disparam enquanto você digita num campo (a nota, por exemplo); `Esc` sai do campo.

## Revisão (Passo 5B)

- **Editor de legenda (palavra a palavra):** na `/revisao` (tecla `E`) e no card do candidato
  na página da fonte ("Corrigir a legenda palavra a palavra"). Lista as palavras do trecho com o
  tempo; a que está tocando fica destacada. Clique numa palavra para corrigir (`Enter` salva,
  `Esc` cancela) — **o tempo é sempre o da transcrição**; ⦸ oculta a palavra; ↶ desfaz a
  correção. A prévia ao vivo atualiza na hora e a prévia fiel/render final passam a usar o texto
  novo (o cache do render é invalidado só para o clip mudado).
- **Correções robustas:** cada correção guarda o texto original da palavra. Se a transcrição for
  regerada e a palavra naquela posição não for mais a mesma, a correção fica **obsoleta**: não é
  aplicada e aparece um aviso no editor (e no card) com o botão "Remover obsoletas". Correções
  feitas antes do 5B (sem original) continuam valendo.
- **Glossário por streamer:** marque **"sempre corrigir assim neste streamer"** ao corrigir, ou
  edite a lista em `/streamers/{id}` (seção "Glossário da legenda"). Troca a palavra inteira,
  sem diferenciar maiúsculas, acentos nem a pontuação colada (que é mantida: `polícia!` →
  `viatura!`). Vale para a prévia ao vivo, o `.ass` do render e as transcrições futuras, sempre
  **antes** das correções clip a clip, e **nunca altera a transcrição salva**. Mudar o glossário
  só invalida os renders cuja legenda muda de fato.
- **Antes de publicar:** na `/revisao`, as caixas **"Conferi música/copyright de fundo"** e
  **"Li os alertas da IA"** (esta só quando a IA levantou alertas), com um aviso fixo: a IA **não
  detecta música com direitos autorais**. Aprovar nunca é bloqueado; só o selo **"pronto para
  publicar"** (aprovado + caixas marcadas) e os filtros **"Prontos"** (status da fila e
  "Mostrar" na página da fonte) dependem delas. Marcar/desmarcar entra no histórico (`Z` desfaz).
- **Calibração a partir das suas decisões:** aprovados = positivos, rejeitados = negativos
  (pulados e sugeridos ficam de fora). Uma regressão logística simples (L2, features
  padronizadas, validação cruzada estratificada 5-fold com semente fixa) sugere pesos não
  negativos para os sinais da curva de interesse e para detecção × IA, e mostra **AUC e
  precisão@N antes/depois** sobre os seus vereditos. Com menos de ~50 revisados ou menos de ~10
  de uma das classes, avisa e **não sugere aplicar**.

  ```powershell
  uv run clipador calibrate --all          # todas as fontes (padrão)
  uv run clipador calibrate --source 3     # só uma fonte
  uv run clipador calibrate --all --json   # saída para scripts
  ```

  Na página da fonte, o painel "Ajustes da detecção" tem o card **"Pesos sugeridos pela
  calibração"** (calculado com todas as fontes): **Aplicar como novo preset** salva o preset em
  `data/presets_calibrados.json` (ele aparece na lista de presets) e roda a reanálise barata
  (só a detecção, sem retranscrever); **Reverter** reanalisa com os parâmetros de antes. Nada é
  aplicado automaticamente. O "antes/depois" usa a mesma aproximação linear da nota de detecção
  (média ponderada do detalhamento por sinal) nos dois lados.

## Facecam automática (Passo 5C)

Ao baixar uma fonte (e ao analisar), se o streamer ainda **não tem facecam**, o Clipador procura
sozinho: pega ~16 quadros espalhados pelo vídeo (640 px, sem os primeiros/últimos segundos), acha
rostos com o OpenCV (**YuNet**, baixado uma vez para `data/models/`; sem internet ou se o
download falhar, usa o **Haar** que já vem no OpenCV 4.x; o OpenCV 5 tirou o Haar, por isso o
projeto fixa `opencv-python-headless<5`), fica com o rosto que aparece **sempre no
mesmo lugar** (personagens do jogo que se mexem são descartados) e expande até as bordas da
câmera (fundo da câmera parado × jogo mexendo, e bordas/molduras fixas). Sem borda clara, usa
uma margem proporcional ao rosto encostada no canto mais próximo. Tudo na CPU, alguns segundos.

- A página do streamer mostra o quadro com a caixa, **"Detectada automaticamente (NN% de
  confiança), ajuste se precisar"** e o botão **Detectar de novo**. Os campos x/y/largura/altura
  (e arrastar sobre o quadro) viram ajuste fino opcional.
- **Corrigir:** mexa nos campos ou arraste e salve. A caixa passa a ser "marcada à mão" e **nunca
  é sobrescrita automaticamente**; "Detectar de novo" pede confirmação antes de substituí-la.
- **Sem facecam** (confiança baixa, abaixo de `CLIPADOR_FACECAM_MIN_CONFIDENCE=0.45`): nada é
  salvo, a página avisa "Facecam não detectada, marque à mão se quiser" e os cortes usam um
  layout sem câmera (o *Facecam embaixo* cai para *Fundo borrado*, com aviso).

```powershell
uv run clipador facecam 3              # fonte #3: imprime caixa/confiança, salva data/work/3/facecam_preview.png
uv run clipador facecam Fulano         # última fonte baixada do streamer (nome ou canal)
uv run clipador facecam 3 --nao-salvar # só mostra
uv run clipador facecam 3 --sobrescrever   # substitui também uma caixa marcada à mão
```

Na prévia, verde = caixa salva, amarelo = rosto, vermelho = tentativa rejeitada. Configuração:
`CLIPADOR_FACECAM_AUTO_DETECT` (padrão `true`), `CLIPADOR_FACECAM_MODEL_DOWNLOAD` (`false` = só
Haar, sem internet), `CLIPADOR_FACECAM_SAMPLES` (16).

## Ao vivo (Passo 5E)

A tela **`/ao-vivo`** grava uma live **enquanto ela acontece** e já sugere os cortes, para você
revisar e postar rápido. Continua valendo: só streamers com autorização confirmada, e **nada é
aprovado nem publicado sozinho**.

1. Cole o link do canal (Twitch, Kick ou YouTube) e clique em **Começar a gravar**. Um processo
   separado (`clipador live-capture`, não ocupa o worker) grava em segmentos (padrão 5 min, sem
   reencodar), reconecta se o stream cair e para sozinho quando a live acaba, quando você clica
   **Parar** ou se sobrar menos de 20 GB no disco.
2. Cada segmento fechado vira um job `live_analyze`: áudio → Whisper → nota com uma **linha de
   base rolante** dos últimos 30 min da sessão (a nota fica comparável ao longo da live) →
   candidatos na linha do tempo da live, sem duplicar os que atravessam dois segmentos. Na
   Twitch o **chat ao vivo** também entra (leitura anônima, sem login).
3. Com o Ollama ligado, a IA avalia os novos candidatos. Quem passa da **nota mínima** da sessão
   (padrão 70) ganha o selo **clipado sozinho** e a prévia fiel entra na fila na frente de tudo.
   O menu "Ao vivo" mostra quantos estão esperando e o navegador avisa (som + notificação,
   se você permitir). Na Revisão, o filtro **Ao vivo** traz esses primeiro.
4. Quando a live termina, os segmentos são juntados num vídeo só e a fonte vira um VOD normal
   (dá para clicar em Analisar para a análise completa).

**Vigiar lives:** no cadastro do streamer, "Vigiar lives" faz o worker checar a cada ~2 min se
um canal dele entrou ao vivo e começar a gravar sozinho (desligado por padrão).

**YouTube desde o início:** marque "Desde o início" (só YouTube permite); nas outras plataformas
grava a partir de agora.

Testar sem internet (passa pelo pipeline real; deixe o worker rodando):

```bash
uv run clipador live --simular data/vods/1/video.mp4 --velocidade 10
uv run clipador live https://www.twitch.tv/canal        # gravar uma live pelo terminal
```

Configurações (`.env`): `CLIPADOR_LIVE_SEGMENT_SECONDS`, `CLIPADOR_LIVE_MIN_SCORE`,
`CLIPADOR_LIVE_MIN_FREE_GB`, `CLIPADOR_LIVE_RETENTION_HOURS` (segmentos analisados que não viraram
corte são apagados depois disso), `CLIPADOR_LIVE_BASELINE_MINUTES`, `CLIPADOR_LIVE_TWITCH_CHAT`.

Medido no M4: cada minuto de live leva ~7 s para ser transcrito e analisado, então a análise
acompanha a live com folga. **Não validado ainda com live real** de cada plataforma (só
simulação); se o yt-dlp não achar o stream, atualize-o (`uv lock --upgrade-package yt-dlp`).

## Requisitos

Roda no **PC com Windows 11 + GPU NVIDIA** e no **Mac com Apple Silicon (M1–M4)**. O app escolhe
sozinho o caminho rápido de cada máquina (`clipador doctor` mostra qual está em uso):

| | PC (Windows + NVIDIA) | Mac (Apple Silicon) |
|---|---|---|
| Transcrição | faster-whisper na GPU (CUDA), `medium` | mlx-whisper na GPU (Metal), `large-v3-turbo` |
| Render | NVENC (`h264_nvenc`) | VideoToolbox (`h264_videotoolbox`) |
| IA local | Ollama na GPU (CUDA) | Ollama na GPU (Metal) |
| Render junto com Whisper/IA | não (dividem os 8 GB de VRAM) | sim (o VideoToolbox usa o motor de mídia) |

Sem GPU (ou num Mac Intel) tudo funciona na CPU, só que mais lento. Forçar um caminho:
`CLIPADOR_WHISPER_BACKEND=faster-whisper|mlx`, `CLIPADOR_RENDER_ENCODER=nvenc|videotoolbox|x264`.

**Medido num MacBook M4 (16 GB):** Whisper `large-v3-turbo` via MLX ~7–9x tempo real
(~7–9 min por hora de VOD; na CPU com faster-whisper ficava ~3,5x); corte final 9:16 de 30 s em
~8 s com VideoToolbox (libx264: ~17 s). Meça na sua máquina com `clipador bench-transcribe` e
`clipador render-bench`.

### Mac

```bash
brew install uv git deno
brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg   # com libass
brew install ollama          # opcional (IA local); depois: ollama serve & ollama pull qwen2.5:7b
```

O ffmpeg do core do Homebrew vem **sem libass** (as legendas falham); use o do tap acima. O
`uv sync` instala o mlx-whisper só no Mac. Restaurar dados de outra máquina:
`docs/RESTAURAR_NO_MAC.md`.

### Windows 11

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
