"""Registra previsões e avalia seus resultados com preços observados."""
from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "dados" / "mercado.db"


def _garantir_tabela(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS historico_previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_previsao TEXT NOT NULL,
            ticker TEXT NOT NULL,
            preco_base REAL NOT NULL,
            direcao_prevista TEXT NOT NULL
                CHECK (direcao_prevista IN ('ALTA', 'QUEDA')),
            confianca REAL NOT NULL,
            data_alvo TEXT NOT NULL,
            preco_alvo_realizado REAL,
            status TEXT NOT NULL DEFAULT 'Pendente'
                CHECK (status IN ('Pendente', 'Acerto', 'Erro'))
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_historico_previsoes_pendentes
        ON historico_previsoes (status, data_alvo)
        """
    )


def salvar_previsao(
    ticker: str,
    preco_base: float,
    direcao_prevista: str,
    confianca: float,
    prazo_dias: int = 5,
) -> int:
    """Registra a previsão e devolve o identificador criado."""
    if not ticker.strip():
        raise ValueError("O ticker não pode ficar vazio.")
    if not math.isfinite(preco_base) or preco_base <= 0:
        raise ValueError("O preço-base deve ser um número finito positivo.")
    if not math.isfinite(confianca) or not 0 <= confianca <= 1:
        raise ValueError("A confiança deve estar entre 0 e 1.")
    if isinstance(prazo_dias, bool) or not isinstance(prazo_dias, int) or prazo_dias < 1:
        raise ValueError("O prazo deve ser um número inteiro positivo de dias.")

    direcao = direcao_prevista.strip().upper()
    if direcao not in {"ALTA", "QUEDA"}:
        raise ValueError("A direção prevista deve ser 'ALTA' ou 'QUEDA'.")

    data_previsao = date.today()
    data_alvo = data_previsao + timedelta(days=prazo_dias)
    with sqlite3.connect(DB_PATH) as connection:
        _garantir_tabela(connection)
        cursor = connection.execute(
            """
            INSERT INTO historico_previsoes (
                data_previsao, ticker, preco_base, direcao_prevista,
                confianca, data_alvo, status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'Pendente')
            """,
            (
                data_previsao.isoformat(),
                ticker.strip().upper(),
                float(preco_base),
                direcao,
                float(confianca),
                data_alvo.isoformat(),
            ),
        )
        return int(cursor.lastrowid)


def auditar_previsoes_passadas() -> dict[str, int]:
    """Avalia previsões vencidas quando já existe preço no pregão-alvo ou seguinte."""
    hoje = date.today().isoformat()
    resultados = {"auditadas": 0, "acertos": 0, "erros": 0, "aguardando_preco": 0}

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _garantir_tabela(connection)
        pendentes = connection.execute(
            """
            SELECT id, ticker, preco_base, direcao_prevista, data_alvo
            FROM historico_previsoes
            WHERE status = 'Pendente' AND date(data_alvo) <= date(?)
            ORDER BY data_alvo, id
            """,
            (hoje,),
        ).fetchall()

        for previsao in pendentes:
            cotacao = connection.execute(
                """
                SELECT fechamento, date(data) AS data_preco
                FROM precos
                WHERE ticker = ?
                    AND date(data) >= date(?)
                    AND date(data) <= date(?)
                    AND fechamento IS NOT NULL
                ORDER BY date(data), rowid DESC
                LIMIT 1
                """,
                (previsao["ticker"], previsao["data_alvo"], hoje),
            ).fetchone()
            if cotacao is None:
                resultados["aguardando_preco"] += 1
                continue

            preco_realizado = float(cotacao["fechamento"])
            if not math.isfinite(preco_realizado) or preco_realizado <= 0:
                resultados["aguardando_preco"] += 1
                continue

            preco_base = float(previsao["preco_base"])
            if previsao["direcao_prevista"] == "ALTA":
                acertou = preco_realizado > preco_base
            else:
                acertou = preco_realizado < preco_base
            status = "Acerto" if acertou else "Erro"
            connection.execute(
                """
                UPDATE historico_previsoes
                SET preco_alvo_realizado = ?, status = ?
                WHERE id = ? AND status = 'Pendente'
                """,
                (preco_realizado, status, previsao["id"]),
            )
            resultados["auditadas"] += 1
            resultados["acertos" if acertou else "erros"] += 1

    return resultados
