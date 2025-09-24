#!/usr/bin/env python3
"""
Script per rigenerare rapidamente il vector store con contenuti multimodali.
Usa la cache esistente invece di riprocessare tutto.
"""

import os
import sys
import shutil
import pickle
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import AppConfig
from app import initialize_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document

def load_multimodal_cache():
    """Carica tutti i documenti dalla cache multimodale."""
    cache_dir = Path("cache/multimodal")
    documents = []

    if not cache_dir.exists():
        print("Cache multimodale non trovata!")
        return []

    pickle_files = list(cache_dir.glob("*.pickle"))
    print(f"Trovati {len(pickle_files)} file cache multimodali")
    print("Inizio caricamento cache multimodale...")

    for i, pickle_file in enumerate(pickle_files, 1):
        try:
            if i % 100 == 0:  # Log ogni 100 documenti
                print(f"Processamento cache multimodale: {i}/{len(pickle_files)} file...")

            with open(pickle_file, 'rb') as f:
                result = pickle.load(f)

                # Crea documento da ImageExtractionResult
                if hasattr(result, 'text') and result.text.strip():
                    doc = Document(
                        page_content=f"[IMMAGINE-{result.image_type.upper()}] {result.text}",
                        metadata={
                            'source': result.metadata.get('source_pdf', 'unknown'),
                            'page': result.metadata.get('page_num', 0),
                            'content_type': 'multimodal_image',
                            'image_type': result.image_type,
                            'confidence': result.confidence,
                            'extraction_method': 'OCR'
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"WARN: Errore caricamento {pickle_file.name}: {e}")

    print(f"OK: Caricati {len(documents)} documenti multimodali dalla cache")
    return documents

def load_text_documents():
    """Carica documenti testuali base dai PDF."""
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    print("Caricamento documenti testuali...")
    print("Configurazione in corso...")
    config = AppConfig.from_env()
    print("Configurazione completata")

    # Disabilita temporaneamente multimodale per caricamento veloce
    original_multimodal = config.enable_multimodal
    config.enable_multimodal = False

    try:
        loader = PyPDFDirectoryLoader(config.pdf_directory, glob="**/*.pdf")
        documents = loader.load()
        print(f"OK: Caricati {len(documents)} documenti testuali")
        return documents
    finally:
        config.enable_multimodal = original_multimodal

def build_vector_store():
    """Costruisce il vector store combinando documenti testuali e multimodali."""

    print("COSTRUZIONE VECTOR STORE COMPLETO")
    print("=" * 50)

    print("FASE 1: Rimozione vector store esistente...")
    # Rimuovi vector store esistente
    vector_store_path = "vector_store_cache"
    if os.path.exists(vector_store_path):
        print("Rimozione vector store esistente...")
        shutil.rmtree(vector_store_path)

    print("FASE 2: Caricamento configurazione...")
    # Carica configurazione
    config = AppConfig.from_env()
    print("Configurazione caricata!")

    print("FASE 3: Caricamento documenti multimodali...")
    # Carica documenti multimodali dalla cache
    multimodal_docs = load_multimodal_cache()

    # Carica documenti testuali (veloce, senza multimodale)
    text_docs = load_text_documents()

    # Combina tutti i documenti
    all_documents = text_docs + multimodal_docs
    print(f"Totale documenti: {len(all_documents)} ({len(text_docs)} testo + {len(multimodal_docs)} multimodale)")

    if not all_documents:
        print("ERRORE: Nessun documento trovato!")
        return False

    # Inizializza embeddings
    print("Inizializzazione embeddings...")
    embeddings = initialize_embeddings()

    # Crea text splitters
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.parent_chunk_size,
        chunk_overlap=config.parent_chunk_overlap,
        add_start_index=True
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_chunk_overlap,
        add_start_index=True
    )

    # Crea vector store
    print("Costruzione vector store...")
    try:
        # Processa a batch per evitare problemi di memoria
        batch_size = 100
        total_batches = (len(all_documents) + batch_size - 1) // batch_size

        print(f"Processamento in {total_batches} batch di {batch_size} documenti...")

        # Inizializza con primo batch
        print(f"Batch 1/{total_batches}: creazione vector store iniziale...")
        first_batch = all_documents[:batch_size]
        vector_store = FAISS.from_documents(first_batch, embeddings)
        store = InMemoryStore()

        # Crea retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            k=config.retriever_k,
            search_type=config.search_type
        )

        # Aggiungi primo batch
        print(f"Aggiunta primo batch al retriever...")
        retriever.add_documents(first_batch, ids=None)

        # Processa batch rimanenti
        for i in range(1, total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_documents))
            batch = all_documents[start_idx:end_idx]

            print(f"Batch {i+1}/{total_batches}: processamento {len(batch)} documenti (documenti {start_idx+1}-{end_idx})...")
            retriever.add_documents(batch, ids=None)
            print(f"Batch {i+1}/{total_batches}: completato! Totale processati: {end_idx}/{len(all_documents)}")

        # Salva vector store
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(os.path.join(vector_store_path, "faiss_index"))

        # Salva docstore
        docstore_path = os.path.join(vector_store_path, "docstore.pkl")
        with open(docstore_path, 'wb') as f:
            pickle.dump(store, f)

        # Salva metadata
        metadata = {
            "total_documents": len(all_documents),
            "text_documents": len(text_docs),
            "multimodal_documents": len(multimodal_docs),
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model
        }

        metadata_path = os.path.join(vector_store_path, "metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("OK: Vector store costruito con successo!")
        print(f"Salvato in: {vector_store_path}")
        print(f"Statistiche:")
        print(f"   - Documenti totali: {len(all_documents)}")
        print(f"   - Documenti testo: {len(text_docs)}")
        print(f"   - Documenti multimodali: {len(multimodal_docs)}")

        return True

    except Exception as e:
        print(f"ERRORE costruzione vector store: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = build_vector_store()
    sys.exit(0 if success else 1)