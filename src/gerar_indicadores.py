"""Gera indicadores técnicos (médias, RSI, volatilidade) para treino de ML usando Pandas puro."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados"
DB_PATH = DATA_DIR / "mercado.db"

def calcular_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def gerar_indicadores() -> None:
    print("Conectando ao banco de dados e lendo os preços...")
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM precos ORDER BY data ASC", conn)
    
    if df.empty:
        print("Nenhum dado encontrado. Execute a coleta primeiro.")
        return

    df['data'] = pd.to_datetime(df['data'])
    frames = []

    print("Calculando indicadores matemáticos por ativo...")
    for ticker, group in df.groupby('ticker'):
        group = group.copy().sort_values('data')
        group['sma_21'] = group['fechamento'].rolling(window=21).mean()
        group['sma_200'] = group['fechamento'].rolling(window=200).mean()
        group['rsi_14'] = calcular_rsi(group['fechamento'], period=14)
        group['retorno_diario'] = group['fechamento'].pct_change()
        group['volatilidade_21d'] = group['retorno_diario'].rolling(window=21).std() * (252 ** 0.5)
        group['retorno_5d'] = group['fechamento'].pct_change(periods=5)
        frames.append(group)

    df_final = pd.concat(frames).dropna()

    print(f"Salvando {len(df_final)} registros enriquecidos...")
    df_final.to_parquet(DATA_DIR / "precos_indicadores.parquet", index=False)
    
    with sqlite3.connect(DB_PATH) as conn:
        df_final.to_sql("precos_indicadores", conn, if_exists="replace", index=False)

    print("Tabela 'precos_indicadores' atualizada com sucesso no banco de dados!")

if __name__ == "__main__":
    gerar_indicadores()
