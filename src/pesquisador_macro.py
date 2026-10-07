"""Lê manchetes macroeconômicas em RSS e imprime um boletim com sentimento básico."""

from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import feedparser
import requests

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
FEEDS: Tuple[Tuple[str, str, str, str, str], ...] = (
    (
        "Brasil",
        "Política Econômica Brasil OR Banco Central Selic OR Risco Fiscal",
        "pt-BR",
        "BR",
        "BR:pt-419",
    ),
    (
        "EUA/Global",
        "Federal Reserve Interest Rates OR US Economy",
        "en-US",
        "US",
        "US:en",
    ),
    (
        "China/Commodities",
        "China Economy OR Iron Ore OR Oil Prices",
        "en-US",
        "US",
        "US:en",
    ),
)
HEADLINES_PER_TOPIC = 5
REQUEST_TIMEOUT_SECONDS = 20
logger = logging.getLogger("pesquisador_macro")

NEGATIVE_KEYWORDS = (
    "crise",
    "queda",
    "tensão",
    "tensao",
    "risco",
    "recessão",
    "recessao",
    "guerra",
    "inflação",
    "inflacao",
    "desemprego",
    "selloff",
    "fall",
    "drop",
    "crisis",
    "risk",
    "war",
    "recession",
    "inflation",
    "default",
    "shock",
    "sanction",
)
POSITIVE_KEYWORDS = (
    "alta",
    "crescimento",
    "recuperação",
    "recuperacao",
    "recorde",
    "acordo",
    "corte de juros",
    "growth",
    "recovery",
    "rally",
    "gain",
    "agreement",
    "surplus",
    "upgrade",
)


def _entry_timestamp(entry: object) -> float:
    published = getattr(entry, "published_parsed", None)
    if published is None:
        return 0.0
    return float(calendar.timegm(published))


def buscar_manchetes(
    topico: str, consulta: str, idioma: str, regiao: str, edicao: str
) -> List[Dict[str, str]]:
    """Busca até cinco manchetes; erros de rede são comunicados ao chamador."""
    response = requests.get(
        GOOGLE_NEWS_RSS,
        params={
            "q": consulta,
            "hl": idioma,
            "gl": regiao,
            "ceid": edicao,
        },
        headers={"User-Agent": "AgenteInvestimentos/1.0 (RSS reader)"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    parsed = feedparser.parse(response.content)
    if parsed.bozo:
        logger.warning("O feed RSS de %s foi analisado com avisos: %s", topico, parsed.bozo_exception)

    entries = sorted(
        parsed.entries,
        key=_entry_timestamp,
        reverse=True,
    )[:HEADLINES_PER_TOPIC]
    headlines: List[Dict[str, str]] = []
    for entry in entries:
        title = str(getattr(entry, "title", "")).strip()
        if not title:
            continue
        published = str(
            getattr(entry, "published", getattr(entry, "updated", "Data não informada"))
        )
        source = getattr(entry, "source", {})
        source_title = str(source.get("title", "")).strip() if source else ""
        headlines.append(
            {
                "title": title,
                "published": published,
                "link": str(getattr(entry, "link", "")).strip(),
                "source": source_title,
            }
        )
    return headlines


def _sentiment_score(headline: str) -> int:
    text = headline.casefold()
    negative = sum(keyword in text for keyword in NEGATIVE_KEYWORDS)
    positive = sum(keyword in text for keyword in POSITIVE_KEYWORDS)
    return positive - negative


def _market_thermometer(headlines: List[Dict[str, str]]) -> str:
    score = sum(_sentiment_score(item["title"]) for item in headlines)
    if score >= 2:
        return "Otimista"
    if score <= -2:
        return "Pessimista"
    return "Cauteloso"


def gerar_boletim() -> Optional[str]:
    """Consulta e imprime o boletim; retorna o termômetro ou None sem feeds."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("=" * 72)
    print("PESQUISADOR MACROECONÔMICO E POLÍTICO — BOLETIM DE MANCHETES")
    print(f"Gerado em: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')}")
    print("=" * 72)

    all_headlines: List[Dict[str, str]] = []
    successful_topics = 0
    for topic, query, language, region, edition in FEEDS:
        print(f"\n{topic}")
        print("-" * len(topic))
        try:
            headlines = buscar_manchetes(topic, query, language, region, edition)
        except requests.RequestException as error:
            print(f"Falha ao consultar o feed: {error}")
            continue

        successful_topics += 1
        if not headlines:
            print("Nenhuma manchete encontrada no feed.")
            continue

        for number, item in enumerate(headlines, start=1):
            source = f" — {item['source']}" if item["source"] else ""
            print(f"{number}. {item['title']}{source}")
            print(f"   Publicada: {item['published']}")
            if item["link"]:
                print(f"   {item['link']}")
        all_headlines.extend(headlines)

    if successful_topics == 0:
        print("\nNenhum feed pôde ser consultado; não foi possível calcular o termômetro.")
        return None

    thermometer = _market_thermometer(all_headlines)
    print("\n" + "=" * 72)
    print(f"TERMÔMETRO INDICATIVO DO MERCADO: {thermometer}")
    print(
        "Classificação heurística baseada apenas em palavras nas manchetes; "
        "não representa análise contextual, previsão ou recomendação de alocação."
    )
    print("Fontes: Google News RSS; manchetes sujeitas a atualização e correção.")
    return thermometer


if __name__ == "__main__":
    gerar_boletim()
