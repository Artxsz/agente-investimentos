"""Motor RAG para ler e consultar políticas de investimento e relatórios."""
import argparse
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ROOT = Path(__file__).resolve().parents[1]
POLITICA_DIR = ROOT / "politica"
VECTOR_DB_PATH = ROOT / "dados" / "vector_db"

def processar_documentos():
    print(f"A procurar documentos na pasta {POLITICA_DIR}...")
    
    documentos = []
    # Lê todos os ficheiros Markdown e TXT na pasta de política
    for filepath in POLITICA_DIR.rglob("*.*"):
        if filepath.suffix in ['.md', '.txt']:
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                documentos.extend(loader.load())
            except Exception as e:
                print(f"Erro ao carregar {filepath.name}: {e}")
    
    if not documentos:
        print("Nenhum ficheiro Markdown/TXT encontrado na pasta de política.")
        return

    print(f"Foram encontrados {len(documentos)} documento(s). A dividir o texto em blocos de memória...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    blocos = text_splitter.split_documents(documentos)

    print("A gerar inteligência vetorial (pode demorar alguns segundos na primeira execução)...")
    # Modelo open-source altamente eficiente
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print('A criar o "cérebro" FAISS...')
    db = FAISS.from_documents(blocos, embeddings)
    db.save_local(str(VECTOR_DB_PATH))
    print(f"Base de conhecimento guardada com sucesso em: {VECTOR_DB_PATH}")

def consultar_base(pergunta: str):
    if not VECTOR_DB_PATH.exists():
        print("Erro: A base vetorial não existe. Execute a ação 'processar' primeiro.")
        return

    # Evita logs desnecessários durante a consulta
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(str(VECTOR_DB_PATH), embeddings, allow_dangerous_deserialization=True)
    
    print(f"\n--- O Agente está a consultar as políticas internas ---")
    print(f"Sua Pergunta: {pergunta}\n")
    
    # Procura os 2 trechos de texto matematicamente mais próximos da pergunta
    resultados = db.similarity_search(pergunta, k=2)
    
    if not resultados:
        print("Nenhuma informação relevante encontrada nas políticas.")
        return

    print("Regras encontradas na base:")
    for i, doc in enumerate(resultados, 1):
        fonte = Path(doc.metadata.get('source', '')).name
        print(f"\n[Regra {i}] Extraída de: {fonte}")
        print(f'"{doc.page_content.strip()}"')

def main():
    parser = argparse.ArgumentParser(description="Motor RAG para Políticas de Investimento")
    parser.add_argument("--acao", choices=["processar", "consultar"], required=True)
    parser.add_argument("--pergunta", type=str, help="A pergunta a fazer à base de conhecimento")
    args = parser.parse_args()
    
    if args.acao == "processar":
        processar_documentos()
    elif args.acao == "consultar":
        if not args.pergunta:
            print("Erro: É necessário fornecer uma --pergunta.")
        else:
            consultar_base(args.pergunta)

if __name__ == "__main__":
    main()
