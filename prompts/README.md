# Prompts do Clipador

Os prompts usados com o LLM local (Ollama) ficam aqui, versionados no git.

| Arquivo | Uso |
|---|---|
| `enrich_v1.md` | Mensagem de sistema do estágio `enrich`: notas, motivo, alertas, títulos, descrição e hashtags. |

## Como editar

- **Ajuste pequeno** (texto, exemplos, ênfase): edite o arquivo atual. A versão usada no cache é
  `nome-do-arquivo + hash do conteúdo`, então qualquer mudança invalida o cache e a próxima
  execução de "Enriquecer com IA" chama o modelo de novo para todos os candidatos.
- **Mudança de comportamento** (critérios novos, formato diferente): crie `enrich_v2.md`, mantenha
  o v1 para comparar, e aponte `CLIPADOR_ENRICH_PROMPT=enrich_v2` no `.env`. Compare os dois com
  `clipador eval <fonte> --golden golden.txt --with-llm` depois de enriquecer com cada um.
- Não coloque o formato JSON no prompt à mão: o schema é enviado pelo código
  (`src/clipador/llm/schema.py`) no parâmetro `format` do Ollama.
- O código monta a mensagem do usuário com os dados do clip. Transcrição e chat vão entre
  marcadores `<<<…_INICIO>>>` / `<<<…_FIM>>>` e esses marcadores são removidos do texto não
  confiável antes de entrar no prompt. Mantenha no prompt a regra de tratar esse texto só como
  dados.
- O crédito ao streamer e os links NÃO vêm do LLM: são adicionados por código no pacote de postagem.
