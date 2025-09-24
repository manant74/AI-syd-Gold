#!/usr/bin/env python3
"""
Script dedicato per costruire il vector store utilizzando il nuovo approccio ibrido.
Usa MultimodalDocumentProcessor come unica fonte per l'estrazione dei dati.
"""

import os
import sys
import shutil
import pickle
import json
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import AppConfig
from app import initialize_embeddings
from extensions.multimodal import MultimodalDocumentProcessor

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

def build_hybrid_store():
    """Costruisce il vector store usando solo il processore ibrido."""

    print("COSTRUZIONE VECTOR STORE IBRIDO")
    print("=" * 50)

    # --- 1. Configurazione e Pulizia ---
    print("FASE 1: Caricamento configurazione e pulizia...")
    config = AppConfig.from_env()
    vector_store_path = Path(config.vector_store_directory)

    if vector_store_path.exists():
        print(f"Rimozione vector store esistente in: {vector_store_path}")
        shutil.rmtree(vector_store_path)
    vector_store_path.mkdir(parents=True, exist_ok=True)

    # --- 2. Estrazione Dati con Logica Ibrida ---
    print("\nFASE 2: Estrazione documenti con MultimodalDocumentProcessor...")
    multimodal_processor = MultimodalDocumentProcessor(config)
    
    # Pulisce la cache delle immagini/ocr per forzare la ri-elaborazione
    print("Pulizia della cache multimodale per una nuova estrazione...")
    multimodal_processor.clear_cache()

    all_documents = []
    pdf_directory = Path(config.pdf_directory)
    pdf_files = list(pdf_directory.glob("**/*.pdf"))

    if not pdf_files:
        print(f"ERRORE: Nessun file PDF trovato in {pdf_directory}")
        return False

    print(f"Trovati {len(pdf_files)} file PDF da processare.")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n--- Processando {i}/{len(pdf_files)}: {pdf_path.name} ---")
        try:
            docs, stats = multimodal_processor.process_pdf_document(str(pdf_path))
            all_documents.extend(docs)
            print(f"OK: Estratti {len(docs)} documenti. Pagine totali: {stats.total_pages}")
        except Exception as e:
            print(f"ERRORE durante il processamento di {pdf_path.name}: {e}")

    if not all_documents:
        print("\nERRORE: Nessun documento è stato estratto. Impossibile costruire l'indice.")
        return False

    print(f"\nEstrazione completata. Totale documenti estratti: {len(all_documents)}")

    # --- 3. Creazione e Salvataggio del Vector Store ---
    print("\nFASE 3: Costruzione del nuovo vector store FAISS...")
    try:
        embeddings = initialize_embeddings()

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.parent_chunk_size)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.child_chunk_size)
        
        vector_store = FAISS.from_documents(all_documents, embeddings)
        store = InMemoryStore()
        
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        print("Aggiunta documenti al retriever...")
        retriever.add_documents(all_documents, ids=None)

        # --- Salvataggio ---
        print("Salvataggio dell'indice e del docstore...")
        faiss_index_path = vector_store_path / "faiss_index"
        docstore_path = vector_store_path / "docstore.pkl"
        metadata_path = vector_store_path / "metadata.json"

        vector_store.save_local(str(faiss_index_path))
        with open(docstore_path, 'wb') as f:
            pickle.dump(store, f)

        metadata = {
            "total_documents": len(all_documents),
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "source": "build_hybrid_store.py"
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\nOK: Vector store ibrido costruito con successo!")
        print(f"Salvato in: {vector_store_path}")
        print(f"Statistiche: {len(all_documents)} documenti totali.")
        return True

    except Exception as e:
        import traceback
        print(f"\nERRORE CRITICO durante la costruzione del vector store: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = build_hybrid_store()
    sys.exit(0 if success else 1)
