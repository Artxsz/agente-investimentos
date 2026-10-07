"""Coleta preços ajustados e séries macroeconômicas para o agente."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

VERSAO = "2.0"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados"
DB_PATH = DATA_DIR / "mercado.db"
BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
BCB_SERIES = {
    "cdi_diario": 12,
    "selic_meta": 432,
    "ipca_mensal": 433,
}
OUTPUT_FILES = (
    "mercado.db",
    "precos.parquet",
    "cdi_diario.parquet",
    "selic_meta.parquet",
    "ipca_mensal.parquet",
    "resumo_ativos.parquet",
    "relatorio_coleta.txt",
)
PRICE_COLUMNS = (
    "data",
    "ticker",
    "abertura",
    "maxima",
    "minima",
    "fechamento",
    "fechamento_ajustado",
    "volume",
)
logger = logging.getLogger("coletar_dados")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Tickers do Yahoo Finance (padrão: ativos já registrados no banco)",
    )
    parser.add_argument(
        "--inicio",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=365),
        help="Data inicial no formato AAAA-MM-DD",
    )
    parser.add_argument(
        "--fim",
        type=date.fromisoformat,
        default=date.today(),
        help="Data final no formato AAAA-MM-DD (padrão: hoje)",
    )
    parser.add_argument(
        "--incluir-hoje",
        action="store_true",
        help="Mantém os preços do pregão de hoje, se já publicados",
    )
    return parser.parse_args()


def existing_tickers() -> list[str]:
    """Reutiliza os tickers da coleta anterior quando não foram passados no CLI."""
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as connection:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='precos'",
            connection,
        )
        if tables.empty:
            return []
        tickers = pd.read_sql("SELECT DISTINCT ticker FROM precos", connection)
    return sorted(tickers["ticker"].dropna().astype(str).tolist())


def collect_bcb_series(
    name: str, code: int, start: date, end: date
) -> pd.DataFrame:
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }
    response = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                BCB_URL.format(code=code),
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            break
        except requests.RequestException as error:
            if attempt == 3:
                raise
            logger.warning(
                "Falha ao consultar série BCB %s (tentativa %d/3): %s. "
                "Nova tentativa em 5 segundos.",
                name,
                attempt,
                error,
            )
            time.sleep(5)

    if response is None:
        raise RuntimeError(f"Não foi possível obter a série BCB {name}.")

    frame = pd.DataFrame(response.json())
    if frame.empty:
        logger.warning("BCB não retornou observações para %s (%s).", name, code)
        return pd.DataFrame(columns=["data", "valor", "serie"])

    frame["data"] = pd.to_datetime(frame["data"], format="%d/%m/%Y")
    frame["valor"] = pd.to_numeric(frame["valor"], errors="coerce")
    frame["serie"] = name
    valid = frame["data"].notna() & np.isfinite(frame["valor"])
    discarded = int((~valid).sum())
    if discarded:
        logger.warning("Série %s: descartadas %d observações inválidas.", name, discarded)
    return frame.loc[valid, ["data", "valor", "serie"]].reset_index(drop=True)


def collect_prices(
    tickers: list[str],
    start: date,
    end: date,
    include_today: bool = False,
) -> pd.DataFrame:
    """Baixa preços ajustados, descarta observações incompletas e alinha calendários."""
    if not tickers:
        raise ValueError("Informe ao menos um ticker para a coleta.")

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=False,
            )
            if history.empty:
                raise RuntimeError("Yahoo Finance não retornou preços.")

            history = history.rename(
                columns={
                    "Open": "abertura",
                    "High": "maxima",
                    "Low": "minima",
                    "Close": "fechamento",
                    "Adj Close": "fechamento_ajustado",
                    "Volume": "volume",
                }
            )
            missing = set(PRICE_COLUMNS[2:]) - set(history.columns)
            if missing:
                raise RuntimeError(
                    "Colunas de preço ausentes: " + ", ".join(sorted(missing))
                )

            index = pd.to_datetime(history.index)
            if index.tz is not None:
                index = index.tz_localize(None)
            history.index = index.normalize()
            history.index.name = "data"
            history = history.reset_index()
            history["ticker"] = ticker
            history = history.loc[:, PRICE_COLUMNS]
            for column in PRICE_COLUMNS[2:]:
                history[column] = pd.to_numeric(history[column], errors="coerce")

            valid = history[list(PRICE_COLUMNS[2:])].notna().all(axis=1)
            valid &= np.isfinite(history[list(PRICE_COLUMNS[2:])]).all(axis=1)
            valid &= history[["abertura", "maxima", "minima", "fechamento",
                              "fechamento_ajustado", "volume"]].gt(0).all(axis=1)
            if not include_today and end == date.today():
                valid &= history["data"].dt.date < date.today()

            discarded = int((~valid).sum())
            if discarded:
                logger.warning(
                    "%s: descartadas %d de %d linhas incompletas, inválidas ou do pregão corrente.",
                    ticker,
                    discarded,
                    len(history),
                )
            clean = history.loc[valid].copy()
            if clean.empty:
                raise RuntimeError("Nenhuma linha válida após a limpeza de preços.")
            frames.append(clean)
            logger.info("%s: %d observações válidas.", ticker, len(clean))
        except Exception as error:
            logger.error("Falha ao coletar %s: %s", ticker, error)
            failures.append(ticker)

    if failures:
        raise RuntimeError(
            "Coleta interrompida; falha para: " + ", ".join(failures)
        )
    if not frames:
        raise RuntimeError("Nenhum preço válido foi coletado.")

    date_sets = [set(frame["data"]) for frame in frames]
    common_dates = set.intersection(*date_sets)
    if not common_dates:
        raise RuntimeError(
            "Os tickers não têm datas válidas em comum; não é possível alinhar calendários."
        )

    common_index = pd.DatetimeIndex(sorted(common_dates))
    for frame in frames:
        omitted = len(frame) - len(common_index)
        if omitted:
            logger.warning(
                "%s: %d datas fora do calendário comum foram excluídas.",
                frame["ticker"].iloc[0],
                omitted,
            )

    aligned = pd.concat(
        [frame[frame["data"].isin(common_index)] for frame in frames],
        ignore_index=True,
    )
    aligned = aligned.sort_values(["data", "ticker"]).reset_index(drop=True)
    logger.info(
        "Calendário alinhado: %d datas comuns para %d ticker(s).",
        len(common_index),
        len(frames),
    )
    return aligned


def _read_existing_table(table: str, keys: list[str]) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=keys)
    with sqlite3.connect(DB_PATH) as connection:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            connection,
            params=(table,),
        )
        if tables.empty:
            return pd.DataFrame(columns=keys)
        return pd.read_sql(f'SELECT * FROM "{table}"', connection)


def _merge_existing(
    new_data: pd.DataFrame, table: str, keys: list[str]
) -> pd.DataFrame:
    old_data = _read_existing_table(table, keys)
    if old_data.empty:
        return new_data.reset_index(drop=True)
    combined = pd.concat([old_data, new_data], ignore_index=True, sort=False)
    combined["data"] = pd.to_datetime(combined["data"], errors="coerce")
    combined = combined.dropna(subset=["data", *keys])
    return (
        combined.drop_duplicates(subset=keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def archive_outputs() -> Path:
    archive_path = DATA_DIR / "mercado.zip"
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for filename in OUTPUT_FILES:
            path = DATA_DIR / filename
            if path.exists():
                archive.write(path, arcname=filename)
    return archive_path


def write_outputs(
    prices: pd.DataFrame,
    macro: dict[str, pd.DataFrame],
    start: date,
    end: date,
) -> None:
    """Atualiza saídas preservando datas e tickers já armazenados fora do intervalo."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_prices = _merge_existing(prices, "precos", ["data", "ticker"])
    all_macro = {
        name: _merge_existing(frame, name, ["data", "serie"])
        for name, frame in macro.items()
    }

    with sqlite3.connect(DB_PATH) as connection:
        all_prices.to_sql("precos", connection, if_exists="replace", index=False)
        for name, frame in all_macro.items():
            frame.to_sql(name, connection, if_exists="replace", index=False)

    all_prices.to_parquet(DATA_DIR / "precos.parquet", index=False)
    for name, frame in all_macro.items():
        frame.to_parquet(DATA_DIR / f"{name}.parquet", index=False)

    summary = (
        all_prices.sort_values("data")
        .groupby("ticker", as_index=False)
        .agg(
            ultima_data=("data", "last"),
            ultimo_fechamento=("fechamento", "last"),
            primeiro_ajustado=("fechamento_ajustado", "first"),
            ultimo_ajustado=("fechamento_ajustado", "last"),
            observacoes=("data", "size"),
        )
    )
    summary["retorno_periodo_pct"] = (
        summary["ultimo_ajustado"] / summary["primeiro_ajustado"] - 1
    ) * 100
    summary.to_parquet(DATA_DIR / "resumo_ativos.parquet", index=False)
    with sqlite3.connect(DB_PATH) as connection:
        summary.to_sql("resumo_ativos", connection, if_exists="replace", index=False)

    lines = [
        f"Versão do coletor: {VERSAO}",
        f"Período solicitado: {start.isoformat()} a {end.isoformat()}",
        f"Período armazenado: {all_prices['data'].min().date()} a {all_prices['data'].max().date()}",
        f"Preços armazenados: {len(all_prices)} registros",
    ]
    lines.extend(
        f"{name}: {len(frame)} registros armazenados"
        for name, frame in all_macro.items()
    )
    (DATA_DIR / "relatorio_coleta.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    archive_outputs()
    logger.info("Dados existentes preservados; atualizadas observações do período solicitado.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args()
    if args.inicio > args.fim:
        raise SystemExit("--inicio deve ser anterior ou igual a --fim.")

    tickers = args.tickers or existing_tickers()
    if not tickers:
        raise SystemExit(
            "Não há tickers no banco. Informe-os explicitamente com --tickers."
        )
    prices = collect_prices(
        tickers,
        args.inicio,
        args.fim,
        include_today=args.incluir_hoje,
    )
    macro = {
        name: collect_bcb_series(name, code, args.inicio, args.fim)
        for name, code in BCB_SERIES.items()
    }
    write_outputs(prices, macro, args.inicio, args.fim)
    print(f"Coleta v{VERSAO} concluída; relatório: {DATA_DIR / 'relatorio_coleta.txt'}")


if __name__ == "__main__":
    main()
