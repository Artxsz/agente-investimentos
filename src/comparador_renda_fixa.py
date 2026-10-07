"""Compara retornos estimados de aplicações (Prefixadas ou % do CDI) buscando a taxa atual do banco."""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "dados" / "mercado.db"

def get_taxa_base(serie_nome: str) -> float:
    """Busca a última taxa anualizada registrada no banco de dados (Selic ou CDI)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT valor FROM {serie_nome} ORDER BY data DESC LIMIT 1")
            resultado = cursor.fetchone()
            if resultado:
                return float(resultado[0])
    except Exception:
        pass
    return 10.50  # Taxa de fallback caso o banco não exista ou esteja vazio

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oferta",
        nargs=4,
        action="append",
        metavar=("NOME", "TIPO", "VALOR", "IR_PCT"),
        required=True,
        help="Ex: 'CDB_BB CDI 110 15' ou 'Tesouro PREFIXADO 12.5 15'",
    )
    parser.add_argument("--dias", type=int, required=True, help="Prazo em dias corridos")
    parser.add_argument("--aporte", type=float, default=1000.0, help="Valor inicial em reais")
    return parser.parse_args()

def compare_offer(
    name: str, tipo: str, valor: float, tax_pct: float, days: int, principal: float, taxa_base: float
) -> dict[str, float | str]:
    
    # Define a taxa anual efetiva com base no tipo da oferta
    if tipo.upper() == "CDI":
        annual_rate_pct = taxa_base * (valor / 100.0)
    elif tipo.upper() == "PREFIXADO":
        annual_rate_pct = valor
    else:
        raise ValueError(f"Tipo desconhecido: {tipo}. Use CDI ou PREFIXADO.")

    gross_factor = (1 + annual_rate_pct / 100) ** (days / 365)
    gross_gain = principal * (gross_factor - 1)
    net_value = principal + gross_gain * (1 - tax_pct / 100)
    
    return {
        "oferta": name,
        "tipo_rentabilidade": f"{valor}% do {tipo}" if tipo == "CDI" else f"{valor}% a.a",
        "taxa_anual_projetada_pct": round(annual_rate_pct, 2),
        "valor_liquido_estimado": round(net_value, 2),
        "retorno_liquido_pct": round((net_value / principal - 1) * 100, 2),
    }

def main() -> None:
    args = parse_args()
    taxa_selic_atual = get_taxa_base("selic_meta")
    
    results = []
    for name, tipo, valor, tax in args.oferta:
        results.append(compare_offer(
            name, tipo, float(valor), float(tax), args.dias, args.aporte, taxa_selic_atual
        ))
        
    results.sort(key=lambda item: item["valor_liquido_estimado"], reverse=True)
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()