"""Otimiza a carteira por mínima variância, respeitando os limites de risco."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "dados" / "mercado.db"

IR_CDI = 0.175
RV_MINIMA = 0.15
RV_MAXIMA = 0.30
LIMITE_CLASSE_RV = 0.15
LIMITE_POR_ATIVO = 0.05
LIMITE_ALTERNATIVOS = 0.05
VOLATILIDADE_MAXIMA = 0.08
QUEDA_MAXIMA = 0.10
DIAS_ANO = 252
RETORNO_DIARIO_ABSURDO = 0.50

ASSET_CLASS_OVERRIDES: dict[str, str] = {}
logger = logging.getLogger("otimizador_carteira")


def buscar_taxa_livre_risco() -> float:
    """Preserva a interface anterior: devolve a última Selic observada em decimal."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            frame = pd.read_sql(
                "SELECT valor FROM selic_meta ORDER BY data DESC LIMIT 1",
                connection,
            )
    except sqlite3.Error as error:
        raise RuntimeError(f"Não foi possível ler a série Selic: {error}") from error
    if frame.empty:
        raise RuntimeError("A tabela selic_meta não contém observações.")
    return float(frame.iloc[0]["valor"]) / 100.0


def _carregar_retornos(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")

    with sqlite3.connect(db_path) as connection:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'",
            connection,
        )["name"].tolist()
        if "precos" not in tables:
            raise RuntimeError("O banco não contém a tabela precos.")
        if "cdi_diario" not in tables:
            raise RuntimeError(
                "O banco não contém cdi_diario; a série CDI é necessária "
                "para modelar a parcela de renda fixa."
            )
        prices = pd.read_sql("SELECT * FROM precos", connection)
        cdi = pd.read_sql("SELECT * FROM cdi_diario", connection)

    required = {"data", "ticker", "fechamento_ajustado"}
    missing = required - set(prices.columns)
    if missing:
        raise RuntimeError(
            "Colunas ausentes em precos: " + ", ".join(sorted(missing))
        )
    prices["data"] = pd.to_datetime(prices["data"], errors="coerce").dt.normalize()
    prices["fechamento_ajustado"] = pd.to_numeric(
        prices["fechamento_ajustado"], errors="coerce"
    )
    valid = (
        prices["data"].notna()
        & prices["ticker"].notna()
        & np.isfinite(prices["fechamento_ajustado"])
        & prices["fechamento_ajustado"].gt(0)
    )
    for column in ("abertura", "maxima", "minima", "volume"):
        if column in prices:
            values = pd.to_numeric(prices[column], errors="coerce")
            valid &= np.isfinite(values) & values.gt(0)
    discarded = int((~valid).sum())
    if discarded:
        logger.warning(
            "Otimizador: descartadas %d linhas de preços incompletas ou inválidas.",
            discarded,
        )
    prices = prices.loc[valid, ["data", "ticker", "fechamento_ajustado"]]
    if prices.empty:
        raise RuntimeError("Não há preços ajustados válidos na tabela precos.")

    prices = prices.drop_duplicates(["data", "ticker"], keep="last")
    price_matrix = (
        prices.pivot(index="data", columns="ticker", values="fechamento_ajustado")
        .sort_index()
        .dropna(axis=0, how="any")
    )
    if price_matrix.shape[1] == 0 or price_matrix.shape[0] < 3:
        raise RuntimeError(
            "Não há pelo menos três datas comuns de preços válidos entre os ativos."
        )
    equity_returns = price_matrix.pct_change(fill_method=None)
    equity_returns = equity_returns.replace([np.inf, -np.inf], np.nan)
    equity_outliers = equity_returns.abs().gt(RETORNO_DIARIO_ABSURDO).any(axis=1)
    if equity_outliers.any():
        logger.warning(
            "Removidos %d dias com retorno de ativo acima de ±%.0f%%.",
            int(equity_outliers.sum()),
            RETORNO_DIARIO_ABSURDO * 100,
        )
        equity_returns = equity_returns.loc[~equity_outliers]
    equity_returns = equity_returns.dropna()

    if not {"data", "valor"}.issubset(cdi.columns):
        raise RuntimeError("A tabela cdi_diario precisa conter data e valor.")
    cdi["data"] = pd.to_datetime(cdi["data"], errors="coerce").dt.normalize()
    cdi["valor"] = pd.to_numeric(cdi["valor"], errors="coerce")
    cdi_valid = cdi["data"].notna() & np.isfinite(cdi["valor"]) & cdi["valor"].ge(0)
    if "serie" in cdi:
        cdi_valid &= cdi["serie"].eq("cdi_diario")
    cdi = (
        cdi.loc[cdi_valid, ["data", "valor"]]
        .drop_duplicates("data", keep="last")
        .set_index("data")
        .sort_index()
    )
    if cdi.empty:
        raise RuntimeError("A tabela cdi_diario não contém observações válidas.")

    # SGS 12 fornece a taxa diária em percentual; o IR é aplicado à taxa usada
    # como retorno diário equivalente da parcela CDI_LIQ.
    cdi_returns = cdi["valor"] * (1.0 - IR_CDI) / 100.0
    cdi_returns.name = "CDI_LIQ"
    returns = equity_returns.join(cdi_returns, how="inner")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    outlier_rows = returns.abs().gt(RETORNO_DIARIO_ABSURDO).any(axis=1)
    if outlier_rows.any():
        logger.warning(
            "Removidos %d dias com retorno total (incluindo CDI) acima de ±%.0f%%.",
            int(outlier_rows.sum()),
            RETORNO_DIARIO_ABSURDO * 100,
        )
        returns = returns.loc[~outlier_rows]
    returns = returns.dropna()
    if len(returns) < 30:
        raise RuntimeError(
            f"Amostra alinhada insuficiente: {len(returns)} retornos; "
            "são necessários ao menos 30."
        )
    logger.info(
        "Retornos alinhados: %d observações, de %s a %s, %d ativos com preços.",
        len(returns),
        returns.index.min().date(),
        returns.index.max().date(),
        len(price_matrix.columns),
    )
    return returns


def _classificar_ativos(tickers: list[str]) -> dict[str, str]:
    classes: dict[str, str] = {}
    for ticker in tickers:
        classe = ASSET_CLASS_OVERRIDES.get(ticker)
        if classe is None:
            # O sufixo 11 é uma heurística para FIIs; use --classe para corrigir
            # ETFs ou outros ativos cujo ticker também termine em 11.
            codigo = ticker.upper().removesuffix(".SA")
            known_classes = {
                "BOVA11": "rv",
                "IMAB11": "renda_fixa",
                "IVVB11": "rv",
                "GOLD11": "alternativos",
                "HASH11": "alternativos",
            }
            classe = known_classes.get(
                codigo, "fii" if codigo.endswith("11") else "rv"
            )
        if classe not in {"rv", "fii", "renda_fixa", "alternativos"}:
            raise ValueError(f"Classe inválida para {ticker}: {classe}")
        classes[ticker] = classe
    return classes


def _pesos_iniciais(classes: dict[str, str], rv_max: float) -> np.ndarray:
    tickers = list(classes)
    variable_names = [
        ticker for ticker in tickers if classes[ticker] in {"rv", "fii"}
    ]
    capacity = sum(
        min(
            LIMITE_CLASSE_RV,
            LIMITE_POR_ATIVO
            * sum(classes[ticker] == classe for ticker in variable_names),
        )
        for classe in ("rv", "fii")
    )
    if capacity + 1e-9 < RV_MINIMA:
        raise RuntimeError(
            "Carteira inviável: os ativos disponíveis, limitados a 5% por ativo "
            "e 15% por classe, não permitem atingir o mínimo de 15% em renda variável. "
            f"Capacidade atual: {capacity:.1%}."
        )
    if rv_max < RV_MINIMA:
        raise RuntimeError("O limite máximo de renda variável ficou abaixo de 15%.")

    weights = np.zeros(len(tickers) + 1, dtype=float)
    class_totals = {"rv": 0.0, "fii": 0.0}
    remaining = RV_MINIMA
    for ticker in variable_names:
        classe = classes[ticker]
        allocation = min(
            LIMITE_POR_ATIVO,
            LIMITE_CLASSE_RV - class_totals[classe],
            remaining,
        )
        if allocation > 0:
            weights[tickers.index(ticker)] = allocation
            class_totals[classe] += allocation
            remaining -= allocation
        if remaining <= 1e-9:
            break
    if remaining > 1e-8:
        raise RuntimeError("Não foi possível construir uma alocação inicial viável.")
    weights[-1] = 1.0 - RV_MINIMA
    return weights


def _otimizar_minima_variancia(
    returns: pd.DataFrame,
    classes: dict[str, str],
    rv_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    tickers = list(classes)
    columns = tickers + ["CDI_LIQ"]
    if list(returns.columns) != columns:
        returns = returns.loc[:, columns]
    if len(returns) < 3:
        raise RuntimeError("A amostra de otimização precisa conter ao menos 3 retornos.")

    estimator = LedoitWolf()
    estimator.fit(returns.to_numpy(dtype=float))
    covariance = estimator.covariance_ * DIAS_ANO
    if not np.isfinite(covariance).all():
        raise RuntimeError("A matriz de covariância contém valores inválidos.")

    rv_indices = [i for i, ticker in enumerate(tickers) if classes[ticker] == "rv"]
    fii_indices = [i for i, ticker in enumerate(tickers) if classes[ticker] == "fii"]
    alternative_indices = [
        i for i, ticker in enumerate(tickers) if classes[ticker] == "alternativos"
    ]
    variable_indices = rv_indices + fii_indices

    def class_weight(weights: np.ndarray, indices: list[int]) -> float:
        return float(weights[indices].sum()) if indices else 0.0

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: class_weight(weights, variable_indices) - RV_MINIMA,
        },
        {
            "type": "ineq",
            "fun": lambda weights: rv_max - class_weight(weights, variable_indices),
        },
        {
            "type": "ineq",
            "fun": lambda weights: LIMITE_CLASSE_RV - class_weight(weights, rv_indices),
        },
        {
            "type": "ineq",
            "fun": lambda weights: LIMITE_CLASSE_RV - class_weight(weights, fii_indices),
        },
        {
            "type": "ineq",
            "fun": lambda weights: LIMITE_ALTERNATIVOS
            - class_weight(weights, alternative_indices),
        },
    ]
    bounds = [
        (
            0.0,
            LIMITE_POR_ATIVO
            if classes[ticker] in {"rv", "fii", "alternativos"}
            else 1.0,
        )
        for ticker in tickers
    ] + [(0.0, 1.0)]

    def variance(weights: np.ndarray) -> float:
        return float(
            np.sum(weights[:, np.newaxis] * covariance * weights[np.newaxis, :])
        )

    result = minimize(
        variance,
        _pesos_iniciais(classes, rv_max),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Otimizador não convergiu: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    return weights, covariance


def _metricas(
    returns: pd.DataFrame, weights: np.ndarray
) -> tuple[float, float]:
    return_matrix = returns.to_numpy(dtype=float)
    if not np.isfinite(return_matrix).all() or not np.isfinite(weights).all():
        raise RuntimeError("Retornos ou pesos não finitos ao calcular métricas.")
    portfolio_returns = np.sum(return_matrix * weights, axis=1)
    annual_volatility = float(np.std(portfolio_returns, ddof=1) * np.sqrt(DIAS_ANO))
    wealth = np.cumprod(1.0 + portfolio_returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    drawdown = wealth / peaks - 1.0
    max_drawdown = float(abs(min(0.0, drawdown.min())))
    return annual_volatility, max_drawdown


def _relatorio_teste_fora_amostra(
    returns: pd.DataFrame, classes: dict[str, str]
) -> float:
    split_index = int(len(returns) * 0.70)
    training = returns.iloc[:split_index]
    test = returns.iloc[split_index:]
    if len(training) < 30 or len(test) < 10:
        raise RuntimeError(
            "Amostra insuficiente para teste temporal 70/30 "
            f"(treino={len(training)}, teste={len(test)})."
        )

    # Se a queda fora da amostra exceder 10%, tenta novamente com tetos menores
    # para a parcela variável; nunca reduz o piso obrigatório de 15%.
    ceilings = (0.30, 0.25, 0.20, 0.15)
    for rv_max in ceilings:
        train_weights, _ = _otimizar_minima_variancia(training, classes, rv_max)
        volatility, drawdown = _metricas(test, train_weights)
        print(
            f"Teste OOS (teto RV {rv_max:.0%}; "
            f"{test.index.min().date()} a {test.index.max().date()}): "
            f"volatilidade {volatility:.2%} a.a.; queda máxima {drawdown:.2%}."
        )
        if volatility < 0.04:
            print(
                "Aviso OOS: volatilidade abaixo da faixa-alvo de 4%-8% a.a.; "
                "o teto rígido de 8% continua respeitado."
            )
        if volatility <= VOLATILIDADE_MAXIMA + 1e-6 and drawdown <= QUEDA_MAXIMA:
            return rv_max
    raise RuntimeError(
        "Teste fora da amostra reprovado: não foi possível manter volatilidade "
        "até 8% a.a. e queda máxima até 10%, mesmo com renda variável no piso de 15%. "
        "A alocação não será apresentada como adequada à política."
    )


def otimizar_carteira(
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Otimiza a carteira; mantém o uso sem argumentos pelo agente unificado."""
    print("A carregar preços ajustados e CDI líquido para mínima variância...")
    returns = _carregar_retornos(Path(db_path) if db_path is not None else DB_PATH)
    asset_classes = _classificar_ativos(
        [ticker for ticker in returns.columns if ticker != "CDI_LIQ"]
    )

    rv_max = _relatorio_teste_fora_amostra(returns, asset_classes)
    weights, covariance = _otimizar_minima_variancia(
        returns, asset_classes, rv_max
    )
    volatility, drawdown = _metricas(returns, weights)

    print("\n--- MÍNIMA VARIÂNCIA COM RESTRIÇÕES DE RISCO ---")
    print(
        f"Dados alinhados: {returns.index.min().date()} a "
        f"{returns.index.max().date()} ({len(returns)} retornos diários)"
    )
    print(f"Covariância: Ledoit-Wolf; CDI líquido com IR assumido de {IR_CDI:.1%}.")
    print(
        f"Renda variável: {sum(weights[i] for i, ticker in enumerate(asset_classes) if asset_classes[ticker] in {'rv', 'fii'}):.2%} "
        f"(limite usado {rv_max:.0%}; faixa permitida 15%-30%)."
    )
    print(f"Volatilidade histórica anualizada: {volatility:.2%} (teto: 8%).")
    print(f"Queda máxima histórica: {drawdown:.2%} (teto: 10%).")

    if volatility > VOLATILIDADE_MAXIMA + 1e-6 or drawdown > QUEDA_MAXIMA:
        raise RuntimeError(
            "A solução de amostra completa viola o teto de risco; "
            "a alocação não será apresentada."
        )

    print("\nALOCAÇÃO SUGERIDA (sujeita à política e às limitações dos dados):")
    names = list(asset_classes) + ["CDI_LIQ"]
    for ticker, weight in zip(names, weights):
        if weight > 0.001:
            classe = "CDI líquido" if ticker == "CDI_LIQ" else asset_classes[ticker]
            print(f"-> {ticker}: {weight:.2%} ({classe})")
    if volatility < 0.04:
        print(
            "Aviso: a volatilidade ficou abaixo da faixa-alvo de 4%-8% a.a.; "
            "o objetivo de mínima variância prioriza menor risco."
        )
    return {
        "weights": dict(zip(names, (float(weight) for weight in weights))),
        "classes": dict(asset_classes),
        "volatility": volatility,
        "max_drawdown": drawdown,
        "rv_ceiling": rv_max,
        "start": returns.index.min().date().isoformat(),
        "end": returns.index.max().date().isoformat(),
        "observations": len(returns),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="dados/mercado.db",
        help="Caminho do SQLite (relativo à raiz do projeto ou absoluto)",
    )
    parser.add_argument(
        "--classe",
        action="append",
        default=[],
        metavar="TICKER=CLASSE",
        help="Classificação explícita: rv, fii ou renda_fixa (repita por ticker)",
    )
    return parser.parse_args()


def main() -> None:
    global ASSET_CLASS_OVERRIDES, DB_PATH
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    db_arg = Path(args.db).expanduser()
    DB_PATH = db_arg if db_arg.is_absolute() else ROOT / db_arg

    overrides: dict[str, str] = {}
    for item in args.classe:
        if "=" not in item:
            raise SystemExit("--classe deve estar no formato TICKER=CLASSE.")
        ticker, classe = item.split("=", 1)
        if classe not in {"rv", "fii", "renda_fixa", "alternativos"}:
            raise SystemExit("CLASSE deve ser rv, fii ou renda_fixa.")
        overrides[ticker] = classe
    ASSET_CLASS_OVERRIDES = overrides

    try:
        otimizar_carteira()
    except (FileNotFoundError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"Erro: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
