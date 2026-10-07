# Agente de investimentos

No terminal, entre primeiro nesta pasta (`cd agente-investimentos`). Crie o
ambiente com `python3 -m venv .venv`, atualize o pip com
`.venv/bin/python -m pip install --upgrade pip` e instale as dependências com
`.venv/bin/python -m pip install -r requirements.txt`. Para coletar dados,
execute `.venv/bin/python src/coletar_dados.py --tickers PETR4.SA VALE3.SA`.
Sem `--tickers`, o coletor reutiliza os ativos já existentes no banco.

Os arquivos da coleta são gerados em `dados/`; eles são saídas locais e não
devem ser enviados ao Projeto do Claude. Cada coleta também cria
`dados/mercado.zip` com o banco, os arquivos Parquet e o relatório. O comparador
aceita ofertas de renda fixa pela linha de comando; consulte `--help` nos
scripts.

Para indexar a política e depois consultá-la:

```bash
.venv/bin/python src/motor_rag.py --acao processar
.venv/bin/python src/motor_rag.py --acao consultar --pergunta "Qual é o limite por ativo?"
```

Na primeira execução, o modelo de embeddings `all-MiniLM-L6-v2` será baixado.
Os documentos usados pelo RAG ficam em `politica/`; a política atual está em
`politica/politica_investimento.md`. Adicione documentos de referência às
pastas de `conhecimento/`.

Para imprimir um boletim macroeconômico a partir das manchetes RSS:

```bash
.venv/bin/python src/pesquisador_macro.py
```

O termômetro usa correspondência simples de palavras-chave; é apenas
informativo e não altera automaticamente a alocação da carteira.

Para abrir o dashboard local:

```bash
.venv/bin/streamlit run src/app_dashboard.py
```

O modelo só mostra a projeção direcional se `dados/modelo_tendencia.joblib` e
os indicadores do ticker estiverem disponíveis. A linha tracejada é uma
visualização ilustrativa da direção classificada e da volatilidade histórica,
não um preço-alvo produzido pelo modelo.

O otimizador usa mínima variância, CDI líquido e limites de risco:

```bash
.venv/bin/python src/otimizador_carteira.py --db dados/mercado.db
```

Cada ativo de renda variável é limitado a 5%; para atingir o mínimo de 15% em
renda variável, são necessários ativos suficientes para respeitar também o
limite de 15% por classe. Tickers terminados em `11` são classificados como
FIIs por padrão; corrija ativos ambíguos, por exemplo, com
`--classe BOVA11.SA=rv` ou `--classe IMAB11.SA=renda_fixa`.

O workflow `.github/workflows/agente_diario.yml` automatiza a coleta e a
execução em dias úteis, além de permitir execução manual pelo GitHub Actions.
Ele requer que o projeto esteja num repositório GitHub com `contents: write`
autorizado para o `GITHUB_TOKEN`; a esteira usa Python 3.11, os tickers
configurados no próprio workflow e treina o modelo antes da previsão diária.
O banco SQLite é adicionado explicitamente apesar de `dados/` estar no
`.gitignore`. A primeira execução da automação começa sem banco no runner e,
por isso, coleta os tickers configurados no workflow.
