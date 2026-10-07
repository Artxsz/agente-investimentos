# Agente de Investimentos Pessoal

Contexto do projeto para o Claude Code. Atualizado em 07/10/2026. Mantenha este arquivo em dia: ao final de cada sessao, atualize as secoes "Estado atual" e "Proximos passos".

## Objetivo

Agente de apoio a decisao para uso pessoal: analisa historico de ativos (acoes, FIIs, ETFs, CDB/CDI/Tesouro) e sugere alocacoes conforme o perfil do investidor. **Nao e consultoria financeira e nao promete previsao de precos.** A decisao final e sempre do investidor.

## Perfil e metas de risco

- Risco **baixo/moderado**. Prioridade: preservar capital e superar o CDI liquido de IR, com ganho real acima da inflacao.
- **Volatilidade da carteira: 4% a 8% a.a. (teto rigido de 8%). Queda maxima tolerada: 10%.**
- **Renda variavel total: 15% a 30%.** Renda fixa: pelo menos 70%.
- Limites: ate 5% por acao, 5% por FII, 15% em acoes+ETFs Brasil somados, 15% em FIIs somados no total da carteira.
- Cripto e metais preciosos: ate 5% somados no total da carteira.
- Sem alavancagem, derivativos, day trade ou venda a descoberto.
- Politica completa em `politica/politica_investimento.md` (v1.1). Faixas e limites sao ilustrativos e ajustaveis pelo investidor.

## Estrutura de pastas

```
agente-investimentos/
├── CLAUDE.md
├── requirements.txt
├── politica/politica_investimento.md
├── src/
│   ├── coletar_dados.py          # coleta BCB + Yahoo, limpa e valida (v2.0)
│   ├── otimizador_carteira.py    # minima variancia com limites, CDI liquido como ativo
│   ├── comparador_renda_fixa.py  # CDB x LCI/LCA x Tesouro Selic liquido de IR
│   ├── pesquisador_macro.py      # manchetes macro RSS e sentimento heuristico
│   ├── modelo_preditivo.py       # classificador direcional de cinco pregoes
│   ├── auditoria_modelo.py       # registro e auditoria futura de previsoes
│   ├── agente.py                 # fluxo executivo de macro, ML, RAG e otimizador
│   └── app_dashboard.py         # dashboard Streamlit com Plotly
├── .github/workflows/
│   └── agente_diario.yml        # coleta e auditoria diária via GitHub Actions
├── dados/                        # saida da coleta (SQLite, Parquet, relatorio_coleta.txt)
├── conhecimento/                 # papers e relatorios (metodologia, macro, valuation)
└── relatorios/                   # relatorios e otimizacoes geradas
```

Execute os scripts sempre a partir da raiz do projeto (`python src/coletar_dados.py`).

## Decisoes ja tomadas

- Dados numericos ficam em arquivos/SQLite locais e sao lidos por codigo; nao vao para a base de conhecimento de texto.
- Fontes: Banco Central (SGS) para Selic, CDI e IPCA; Yahoo Finance (yfinance, tickers `.SA`) para precos e dividendos; CVM e B3 para confirmar dados importantes. Yahoo e fonte nao oficial.
- **Otimizacao: minima variancia com restricoes. NAO maximizar Sharpe com retornos historicos** (gerou 76% em PETR4 e "retorno esperado" de 60% a.a., irreal e fora da politica).
- **CDI liquido e ativo alocavel.** Como tem variancia ~0, minima variancia pura iria 100% para ele; por isso as faixas de renda variavel (15% a 30%) entram como restricoes e o otimizador decide como montar a parte de risco.
- Queda maxima acima de 10% na amostra: reduzir o teto de renda variavel e otimizar de novo. Sempre mostrar tambem o teste fora da amostra.
- Previsao de preco por ARIMA/ML so como baseline a ser superado, nunca como motor de recomendacao.
- Armazenamento simples (Parquet + SQLite/DuckDB). Sem Docker/MongoDB/banco vetorial sem necessidade real.
- **Pendente de confirmacao do investidor:** limite por acao individual. A politica adota 5%; outra IA sugeriu 15%. Parametro `--acao-max` no otimizador.

## Regras para o Claude Code

1. Todo dado numerico citado deve ter fonte e data. Nunca inventar dados; se faltar, dizer o que falta.
2. Retornos sempre **liquidos** de IR e custos, comparados ao **CDI liquido**.
3. Backtests: walk-forward, sem look-ahead bias, sem survivorship bias, com custos. Resultado bom demais (Sharpe sustentado > 1, retorno esperado muito acima do CDI) = suspeito.
4. Antes de sugerir compra ou venda, checar a politica e dizer qual regra foi aplicada. Mostrar riscos e o contraponto.
5. Nao armazenar nem pedir senhas, CPF, extratos ou dados de contas. Carteira real: apenas ativos e proporcoes.
6. Descartar linhas de preco incompletas (abertura/maxima/minima/volume zerados) antes de calcular.
7. Nao sobrescrever arquivos de `dados/` sem avisar. Usar Git; commits pequenos.
8. Responder em portugues.

## Estado atual (07/10/2026)

- `coletar_dados.py` v2.0: usa preços ajustados, descarta linhas incompletas e dias fora do calendário comum, registra qualidade e tenta até três vezes chamadas BCB com timeout de 60 s. Retry validado por teste simulado; coleta online completa não foi repetida nesta sessão.
- `otimizador_carteira.py`: mínima variância com CDI líquido, Ledoit-Wolf, tetos de volatilidade/drawdown e teste OOS temporal; mantém `otimizar_carteira()` sem argumentos e retorna métricas/pesos para consumidores programáticos. Foi executado no banco local; a execução produziu avisos RuntimeWarning internos do SciPy e OOS abaixo da faixa-alvo de volatilidade, mas dentro dos tetos rígidos. BOVA11/IVVB11 são RV, IMAB11 renda fixa e GOLD11/HASH11 alternativos com limite conjunto de 5%.
- `pesquisador_macro.py`: lê até cinco manchetes recentes por tema do Google News RSS e calcula um termômetro heurístico; o boletim já é chamado pelo `agente.py`. Não altera automaticamente a alocação.
- `app_dashboard.py`: dashboard Streamlit com seletor de ticker, termômetro, candles de aproximadamente 180 dias e horizontes visuais de 5/21/42 pregões. O ML só classifica direção para 5 pregões; 21/42 são extrapolações da inclinação da SMA 21, explicitamente identificadas como estatísticas. Helpers e controles dos três horizontes foram validados na página local.
- `auditoria_modelo.py`: cria `historico_previsoes`, registra as previsões executadas pelo agente e avalia as vencidas usando o primeiro fechamento disponível na data-alvo ou depois dela. O prazo padrão é de cinco dias corridos. A avaliação registra acerto/erro; não retreina nem altera o modelo.
- `.github/workflows/agente_diario.yml`: workflow agendado de segunda a sexta às 22:00 UTC e manual. Usa Python 3.11, coleta os tickers configurados, gera indicadores, treina o classificador antes da previsão e força o versionamento de `dados/mercado.db`, ignorado pelo `.gitignore`.
- Um zip de teste anterior (gerado por outro script) tinha so PETR4 e VALE3 por 1 ano, sem dividendos e com a ultima linha de cada ativo incompleta. Nao usar como base.
- Cenario de referencia: Selic meta 13,75% a.a. desde 16/09/2026; CDI ~13,65% a.a.; expectativa de IPCA 2026 ~4,9% (Focus). Reconfirmar antes de decidir.
- Pontos a conferir: IPCA de agosto/2026 apareceu negativo (-0,32%) e a PETR4 teve alta de ~8% em 05/10/2026. Verificar no IBGE e em noticias.
- Se o investidor tiver um `otimizador_carteira.py` anterior (de outra IA), comparar interfaces antes de substituir.

## Proximos passos

1. Rodar `python src/coletar_dados.py --inicio 2019-01-01` e ler `dados/relatorio_coleta.txt`; corrigir o que aparecer.
2. Validar dividendos de FIIs e dados do IPCA em fontes oficiais.
3. Rodar `python src/otimizador_carteira.py --db dados/mercado.db`; conferir vol, drawdown e teste fora da amostra.
4. Incluir ETF de IPCA+ (ex.: IMAB11) na coleta para ter renda fixa marcada a mercado alem do CDI.
5. Analise de FIIs: dividend yield 12m, P/VP, vacancia, consistencia de rendimentos, concentracao.
6. Backtest walk-forward da carteira contra CDI liquido e IFIX/Ibovespa; relatorio periodico no formato da politica (secao 13).
7. Publicar o projeto num repositório GitHub e validar a execução manual do workflow; validar a auditoria de previsões com observações reais após os prazos e adicionar testes automatizados. Investigar os avisos internos do SciPy durante a covariância e manter a projeção do dashboard claramente separada de preço-alvo do modelo.
