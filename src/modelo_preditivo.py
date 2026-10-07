"""Treina um modelo de Machine Learning e faz previsões de tendência."""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados"
DB_PATH = DATA_DIR / "mercado.db"
MODEL_PATH = DATA_DIR / "modelo_tendencia.joblib"

def preparar_dados(ticker: str):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM precos_indicadores WHERE ticker = '{ticker}' ORDER BY data ASC", conn)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker}.")

    df['data'] = pd.to_datetime(df['data'])
    
    # Criar a variável ALVO (Target): O preço vai estar maior daqui a 5 dias?
    df['preco_futuro'] = df['fechamento'].shift(-5)
    df['alvo_alta'] = (df['preco_futuro'] > df['fechamento']).astype(int)
    
    # Features (variáveis) para tentar adivinhar o alvo
    features = ['sma_21', 'sma_200', 'rsi_14', 'volatilidade_21d', 'retorno_5d']
    
    # Remover linhas onde não temos o preço futuro ou falta indicador
    df_treino = df.dropna(subset=features + ['preco_futuro'])
    df_previsao = df.tail(1) # O último dia da base
    
    return df_treino, df_previsao, features

def treinar(ticker: str):
    print(f"Preparando dados para treinar o modelo de {ticker}...")
    df_treino, _, features = preparar_dados(ticker)
    
    X = df_treino[features]
    y = df_treino['alvo_alta']
    
    # Divisão cronológica: 80% treino, 20% teste
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Treino do Random Forest
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)
    
    previsoes = modelo.predict(X_test)
    acc = accuracy_score(y_test, previsoes)
    
    print(f"\n--- Resultados do Backtest ({ticker}) ---")
    print(f"Precisão Global (Accuracy): {acc * 100:.2f}%")
    print("Relatório detalhado:")
    print(classification_report(y_test, previsoes, target_names=['Baixa/Estável', 'Alta']))
    
    # Guardar o modelo
    joblib.dump(modelo, MODEL_PATH)
    print(f"\nModelo guardado com sucesso em: {MODEL_PATH}")

def prever(ticker: str) -> dict[str, str | float | int] | None:
    if not MODEL_PATH.exists():
        print("Erro: Modelo não encontrado. Execute o treino primeiro.")
        return None
        
    _, df_previsao, features = preparar_dados(ticker)
    modelo = joblib.load(MODEL_PATH)
    
    X_novo = df_previsao[features]
    tendencia = modelo.predict(X_novo)[0]
    probabilidade = modelo.predict_proba(X_novo)[0]

    data_atual = df_previsao['data'].iloc[0].strftime('%Y-%m-%d')
    fechamento_atual = df_previsao['fechamento'].iloc[0]
    indice_classe = int(tendencia)
    direcao = "ALTA" if indice_classe == 1 else "QUEDA"
    confianca = float(probabilidade[indice_classe])
    
    print(f"\n--- Previsão do Agente para {ticker} ---")
    print(f"Data Base: {data_atual}")
    print(f"Preço Atual: R$ {fechamento_atual:.2f}")
    
    if tendencia == 1:
        print(f"Previsão da IA: ALTA nos próximos 5 dias.")
        print(f"Confiança matemática: {confianca * 100:.2f}%")
    else:
        print(f"Previsão da IA: QUEDA ou LATERALIZAÇÃO nos próximos 5 dias.")
        print(f"Confiança matemática: {confianca * 100:.2f}%")
    return {
        "ticker": ticker,
        "preco_base": float(fechamento_atual),
        "direcao_prevista": direcao,
        "confianca": confianca,
        "data_base": data_atual,
    }

def main():
    parser = argparse.ArgumentParser(description="Treino e previsão do Agente de ML")
    parser.add_argument("--acao", choices=["treinar", "prever"], required=True)
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    
    if args.acao == "treinar":
        treinar(args.ticker)
    else:
        prever(args.ticker)

if __name__ == "__main__":
    main()
