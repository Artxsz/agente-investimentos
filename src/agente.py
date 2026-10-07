"""Agente de Investimentos Unificado - Relatório Executivo."""
from __future__ import annotations
import sys
import argparse
import sqlite3
from pathlib import Path

# Adiciona a pasta src ao sistema para permitir importar os outros guiões
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import modelo_preditivo
import motor_rag
import otimizador_carteira
import pesquisador_macro
import auditoria_modelo

def painel_agente(ticker: str):
    try:
        auditoria = auditoria_modelo.auditar_previsoes_passadas()
        print(
            "Auditoria de previsões: "
            f"{auditoria['auditadas']} avaliadas, "
            f"{auditoria['acertos']} acertos, {auditoria['erros']} erros, "
            f"{auditoria['aguardando_preco']} aguardando cotação."
        )
    except (sqlite3.Error, OSError) as e:
        print(f"Aviso (Auditoria de previsões): {e}")

    print("="*70)
    print(f"🤖 AGENTE FINANCEIRO DE IA - RELATÓRIO EXECUTIVO PARA {ticker}")
    print("="*70)

    print("\n[PASSO 0] A LER O CENÁRIO MACROECONÓMICO E POLÍTICO...")
    try:
        pesquisador_macro.gerar_boletim()
    except Exception as e:
        print(f"Aviso (Pesquisador Macro): {e}")
    
    # 1. Módulo Quantitativo (Previsão de Preço)
    print("\n[PASSO 1] A CONSULTAR O CÉREBRO MATEMÁTICO (MACHINE LEARNING)...")
    try:
        previsao = modelo_preditivo.prever(ticker)
    except Exception as e:
        print(f"Aviso (Machine Learning): {e}")
    else:
        if previsao is not None:
            try:
                previsao_id = auditoria_modelo.salvar_previsao(
                    ticker=ticker,
                    preco_base=float(previsao["preco_base"]),
                    direcao_prevista=str(previsao["direcao_prevista"]),
                    confianca=float(previsao["confianca"]),
                )
                print(f"Previsão registrada para auditoria (ID {previsao_id}).")
            except (sqlite3.Error, OSError, ValueError, TypeError) as e:
                print(f"Aviso (Registro da previsão): {e}")
        
    print("\n" + "-"*70)
    
    # 2. Módulo de Compliance e Risco (RAG)
    print("[PASSO 2] A VERIFICAR REGRAS DE RISCO (COMPLIANCE)...")
    pergunta_compliance = f"Quais são os limites de concentração, regras ou políticas de investimento para {ticker} ou para o seu setor?"
    try:
        motor_rag.consultar_base(pergunta_compliance)
    except Exception as e:
        print(f"Aviso (Motor RAG): {e}")
        
    print("\n" + "-"*70)
    
    # 3. Módulo de Alocação (Markowitz)
    print("[PASSO 3] A OTIMIZAR A CARTEIRA GLOBAL (MARKOWITZ)...")
    try:
        otimizador_carteira.otimizar_carteira()
    except Exception as e:
        print(f"Aviso (Otimizador): {e}")
        
    print("\n" + "="*70)
    print("✅ ANÁLISE CONCLUÍDA. AGUARDO AS SUAS ORDENS.")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Executa o Agente Financeiro Completo")
    # PETR4.SA será o padrão se o utilizador não digitar nada
    parser.add_argument("--ticker", type=str, default="PETR4.SA", help="O ticker principal a ser analisado (ex: VALE3.SA)")
    args = parser.parse_args()
    
    painel_agente(args.ticker)

if __name__ == "__main__":
    main()
