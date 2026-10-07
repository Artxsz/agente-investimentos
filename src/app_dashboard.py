"""Dashboard Streamlit para visualizar o boletim macro, preço e alocação."""

from __future__ import annotations

import io
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DB_PATH = ROOT / "dados" / "mercado.db"
MODEL_PATH = ROOT / "dados" / "modelo_tendencia.joblib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import modelo_preditivo
import otimizador_carteira
import pesquisador_macro


@st.cache_data(ttl=60)
def listar_tickers(db_path: str) -> list[str]:
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='precos'",
            connection,
        )
        if tables.empty:
            return []
        frame = pd.read_sql(
            "SELECT DISTINCT ticker FROM precos WHERE ticker IS NOT NULL ORDER BY ticker",
            connection,
        )
    return frame["ticker"].astype(str).tolist()


def carregar_precos(ticker: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql(
            """
            SELECT data, abertura, maxima, minima, fechamento
            FROM precos
            WHERE ticker = ?
            ORDER BY data DESC
            """,
            connection,
            params=(ticker,),
        )
    if frame.empty:
        raise ValueError(f"Não há preços disponíveis para {ticker}.")
    required = {"data", "abertura", "maxima", "minima", "fechamento"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Colunas OHLC ausentes: " + ", ".join(sorted(missing))
        )
    frame["data"] = pd.to_datetime(frame["data"], errors="coerce")
    numeric = ["abertura", "maxima", "minima", "fechamento"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["data", *numeric]
    )
    frame = frame.sort_values("data").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Os últimos preços de {ticker} não contêm OHLC válido.")
    latest_date = frame["data"].max()
    six_months_ago = latest_date - pd.Timedelta(days=180)
    frame = frame.loc[frame["data"] >= six_months_ago]
    if frame.empty:
        raise ValueError(f"Não há preços nos últimos seis meses para {ticker}.")
    return frame.reset_index(drop=True)


def prever_tendencia(ticker: str) -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Treine-o antes com "
            "`python src/modelo_preditivo.py --acao treinar --ticker TICKER`."
        )
    _, forecast_frame, features = modelo_preditivo.preparar_dados(ticker)
    if forecast_frame.empty:
        raise ValueError(f"Sem linha recente para prever {ticker}.")
    observation = forecast_frame.loc[:, features]
    if observation.isna().any(axis=None):
        raise ValueError(
            f"Faltam indicadores para formar a previsão de {ticker}; "
            "gere os indicadores novamente."
        )

    model = joblib.load(MODEL_PATH)
    trend = int(model.predict(observation)[0])
    probabilities = model.predict_proba(observation)[0]
    classes = list(model.classes_)
    predicted_index = classes.index(trend)
    probability = float(probabilities[predicted_index])
    return {
        "trend": trend,
        "probability": probability,
        "date": pd.Timestamp(forecast_frame["data"].iloc[0]),
        "close": float(forecast_frame["fechamento"].iloc[0]),
        "volatility_annual": float(
            forecast_frame["volatilidade_21d"].iloc[0]
        ),
    }


def criar_grafico_preco(
    ticker: str,
    prices: pd.DataFrame,
    prediction: Dict[str, Any],
    horizon: str = "1 Semana (5 dias úteis)",
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=prices["data"],
                open=prices["abertura"],
                high=prices["maxima"],
                low=prices["minima"],
                close=prices["fechamento"],
                name=ticker,
            )
        ]
    )

    last_date = pd.Timestamp(prices["data"].iloc[-1])
    last_close = float(prices["fechamento"].iloc[-1])
    horizon_days = {
        "1 Semana (5 dias úteis)": 5,
        "1 Mês (21 dias úteis)": 21,
        "2 Meses (42 dias úteis)": 42,
    }[horizon]
    future_date = last_date + pd.offsets.BDay(horizon_days)

    if horizon_days == 5:
        direction = 1 if prediction["trend"] == 1 else -1
        annual_volatility = max(float(prediction["volatility_annual"]), 0.0)
        illustrative_move = annual_volatility / np.sqrt(252) * np.sqrt(horizon_days)
        projected_close = last_close * (1 + direction * illustrative_move)
        direction_label = "Alta" if direction > 0 else "Baixa/Lateral"
        forecast_name = f"Previsão direcional ML ({direction_label}, 5 dias)"
    else:
        closes = pd.to_numeric(prices["fechamento"], errors="coerce")
        sma_21 = closes.rolling(window=21).mean().dropna()
        if len(sma_21) < 2:
            raise ValueError(
                "São necessários pelo menos 21 fechamentos válidos para "
                "extrapolar a média móvel de 21 dias."
            )
        slope_window = sma_21.tail(21)
        slope = float(
            np.polyfit(np.arange(len(slope_window), dtype=float), slope_window, 1)[0]
        )
        projected_close = last_close + slope * horizon_days
        direction_label = "alta" if slope >= 0 else "baixa"
        forecast_name = (
            f"Extrapolação da inclinação SMA 21 ({direction_label}, "
            f"{horizon_days} dias úteis)"
        )

    fig.add_trace(
        go.Scatter(
            x=[last_date, future_date],
            y=[last_close, projected_close],
            mode="lines+markers",
            name=forecast_name,
            line={"dash": "dash", "width": 2, "color": "#f0a202"},
            hovertemplate="%{x|%Y-%m-%d}<br>Projeção visual: R$ %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{ticker} — histórico de 6 meses e projeção visual de {horizon_days} dias úteis",
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=560,
    )
    return fig


def agrupar_alocacao(allocation: Dict[str, Any]) -> pd.DataFrame:
    weights = allocation["weights"]
    classes = allocation["classes"]
    grouped: Dict[str, float] = {}
    for asset, weight in weights.items():
        if asset == "CDI_LIQ":
            category = "CDI líquido"
        else:
            asset_class = classes.get(asset, "outro")
            category = {
                "rv": "Ações/ETFs",
                "fii": "FIIs",
                "renda_fixa": "Renda fixa",
                "alternativos": "Cripto/metais",
            }.get(asset_class, "Outros")
        grouped[category] = grouped.get(category, 0.0) + float(weight)
    return pd.DataFrame(
        [{"classe": category, "peso": weight} for category, weight in grouped.items()]
        if grouped
        else [],
        columns=["classe", "peso"],
    )


def criar_grafico_alocacao(allocation: Dict[str, Any]) -> go.Figure:
    grouped = agrupar_alocacao(allocation)
    if grouped.empty:
        raise ValueError("O otimizador não retornou pesos para exibir.")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=grouped["classe"],
                values=grouped["peso"],
                hole=0.55,
                texttemplate="%{label}<br>%{percent}",
                hovertemplate="%{label}: %{value:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title="Alocação sugerida por classe", height=450)
    return fig


def executar_analise(ticker: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    macro_output = io.StringIO()
    try:
        with redirect_stdout(macro_output):
            results["thermometer"] = pesquisador_macro.gerar_boletim()
        results["macro_report"] = macro_output.getvalue()
    except Exception as error:
        results["macro_error"] = str(error)
        results["macro_report"] = macro_output.getvalue()

    try:
        results["prices"] = carregar_precos(ticker)
    except Exception as error:
        results["price_error"] = str(error)

    try:
        results["prediction"] = prever_tendencia(ticker)
    except Exception as error:
        results["prediction_error"] = str(error)

    try:
        results["allocation"] = otimizador_carteira.otimizar_carteira(DB_PATH)
    except Exception as error:
        results["allocation_error"] = str(error)
    return results


def main() -> None:
    st.set_page_config(
        page_title="Agente de Investimentos",
        page_icon="📊",
        layout="wide",
    )
    st.title("Agente de Investimentos — Painel executivo")
    st.caption(
        "Ferramenta informativa baseada em dados históricos e manchetes. "
        "Não é consultoria financeira nem promessa de retorno."
    )

    with st.sidebar:
        st.header("Análise")
        tickers = listar_tickers(str(DB_PATH))
        if tickers:
            default_index = tickers.index("PETR4.SA") if "PETR4.SA" in tickers else 0
            ticker = st.selectbox("Ticker", tickers, index=default_index)
        else:
            ticker = st.text_input("Ticker", "PETR4.SA")
            st.warning("Banco ausente ou sem tickers; confirme os dados locais.")
        run_analysis = st.button(
            "Executar Análise da IA",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        with st.spinner("Consultando notícias, modelo e otimizador..."):
            st.session_state["analysis_results"] = executar_analise(ticker)
            st.session_state["analysis_ticker"] = ticker

    results = st.session_state.get("analysis_results")
    if not results:
        st.info("Escolha um ticker e clique em “Executar Análise da IA”.")
        return

    analyzed_ticker = st.session_state.get("analysis_ticker", ticker)
    st.header("Cenário macroeconômico")
    if results.get("thermometer"):
        st.metric("Termômetro do mercado", results["thermometer"])
    elif results.get("macro_error"):
        st.warning(f"Não foi possível consultar as manchetes: {results['macro_error']}")
    else:
        st.info("O termômetro não está disponível porque nenhum feed foi obtido.")
    if results.get("macro_report"):
        with st.expander("Boletim de manchetes"):
            st.text(results["macro_report"])

    st.header(f"Preço e tendência — {analyzed_ticker}")
    horizon = st.selectbox(
        "Horizonte de Projeção",
        [
            "1 Semana (5 dias úteis)",
            "1 Mês (21 dias úteis)",
            "2 Meses (42 dias úteis)",
        ],
        key="projection_horizon",
    )
    if horizon != "1 Semana (5 dias úteis)":
        st.info(
            "Nota: A previsão para 1 e 2 meses é uma extrapolação estatística de tendência. "
            "O motor de Machine Learning está atualmente otimizado para o curto prazo (5 dias)."
        )
    left, right = st.columns([3, 1])
    with left:
        if results.get("price_error"):
            st.error(results["price_error"])
        elif results.get("prices") is not None:
            if results.get("prediction") is not None:
                st.plotly_chart(
                    criar_grafico_preco(
                        analyzed_ticker,
                        results["prices"],
                        results["prediction"],
                        horizon,
                    ),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    go.Figure(
                        data=[
                            go.Candlestick(
                                x=results["prices"]["data"],
                                open=results["prices"]["abertura"],
                                high=results["prices"]["maxima"],
                                low=results["prices"]["minima"],
                                close=results["prices"]["fechamento"],
                                name=analyzed_ticker,
                            )
                        ]
                    ).update_layout(
                        title=f"{analyzed_ticker} — histórico de 6 meses",
                        xaxis_rangeslider_visible=False,
                    ),
                    use_container_width=True,
                )
                st.warning(
                    "Sem previsão do modelo; exibindo apenas os preços históricos."
                )
    with right:
        if results.get("prediction") is not None:
            prediction = results["prediction"]
            trend_label = "Alta" if prediction["trend"] == 1 else "Baixa/Lateral"
            st.metric("Tendência estimada (5 pregões)", trend_label)
            st.metric("Confiança do classificador", f"{prediction['probability']:.1%}")
            st.caption(
                "A linha tracejada é apenas uma ilustração direcional. "
                "A amplitude vem da volatilidade histórica e não é preço-alvo do modelo."
            )
        elif results.get("prediction_error"):
            st.warning(f"Previsão indisponível: {results['prediction_error']}")

    st.header("Alocação de carteira")
    if results.get("allocation") is not None:
        allocation = results["allocation"]
        st.plotly_chart(
            criar_grafico_alocacao(allocation),
            use_container_width=True,
        )
        st.caption(
            f"Volatilidade histórica: {allocation['volatility']:.2%} a.a. · "
            f"Queda máxima: {allocation['max_drawdown']:.2%} · "
            f"Dados: {allocation['start']} a {allocation['end']}."
        )
        grouped = agrupar_alocacao(allocation)
        st.dataframe(
            grouped.assign(peso=grouped["peso"].map(lambda value: f"{value:.2%}")),
            hide_index=True,
            use_container_width=True,
        )
    elif results.get("allocation_error"):
        st.error(f"Alocação indisponível: {results['allocation_error']}")

    st.warning(
        "As previsões classificam tendência e a linha projetada é ilustrativa. "
        "A alocação usa dados históricos; confira a política de risco e valide "
        "as fontes antes de qualquer decisão."
    )


if __name__ == "__main__":
    main()
