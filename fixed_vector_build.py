#!/usr/bin/env python3
"""
Script per costruire il vector store completo con ParentDocumentRetriever.
"""

import os
import sys
import shutil
import pickle
from pathlib import Path

# Configurazione diretta
os.environ["EMBEDDING_PROVIDER"] = "huggingface"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

def load_multimodal_cache():
    """Carica documenti multimodali dalla cache."""
    cache_dir = Path("cache/multimodal")
    documents = []

    if not cache_dir.exists():
        print("ERRORE: Cache multimodale non trovata!")
        return []

    # Prendi solo un subset per velocità (primi 200 documenti)
    pickle_files = list(cache_dir.glob("*.pickle"))[:200]
    print(f"Caricamento {len(pickle_files)} documenti multimodali...")

    for i, pickle_file in enumerate(pickle_files, 1):
        if i % 50 == 0:
            print(f"Processamento cache: {i}/{len(pickle_files)} file...")

        try:
            with open(pickle_file, 'rb') as f:
                result = pickle.load(f)

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
    """Carica alcuni documenti testuali dai PDF."""
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    print("Caricamento documenti testuali...")
    loader = PyPDFDirectoryLoader("pdfs", glob="**/*.pdf")
    documents = loader.load()

    # Prendi solo primi 50 documenti testuali per velocità
    documents = documents[:50] if len(documents) > 50 else documents
    print(f"OK: Caricati {len(documents)} documenti testuali")
    return documents

def main():
    print("COSTRUZIONE VECTOR STORE COMPLETO")
    print("=" * 50)

    # Rimuovi vector store esistente
    vector_store_path = "vector_store_cache"
    if os.path.exists(vector_store_path):
        print("Rimozione vector store esistente...")
        shutil.rmtree(vector_store_path)

    # Carica documenti
    print("FASE 1: Caricamento documenti multimodali...")
    multimodal_docs = load_multimodal_cache()

    print("FASE 2: Caricamento documenti testuali...")
    text_docs = load_text_documents()

    # Combina documenti
    all_documents = text_docs + multimodal_docs
    print(f"Totale documenti: {len(all_documents)} ({len(text_docs)} testo + {len(multimodal_docs)} multimodale)")

    if not all_documents:
        print("ERRORE: Nessun documento trovato!")
        return False

    # Inizializza embeddings base (senza ottimizzazioni)
    print("FASE 3: Inizializzazione HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embeddings inizializzati!")

    # Crea splitters
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )

    # Costruzione vector store con batch piccoli
    print("FASE 4: Costruzione vector store...")
    try:
        # Primo batch per inizializzare
        batch_size = 20  # Batch molto piccoli per evitare problemi memoria
        first_batch = all_documents[:batch_size]

        print(f"Creazione vector store iniziale con {len(first_batch)} documenti...")
        vector_store = FAISS.from_documents(first_batch, embeddings)
        store = InMemoryStore()

        # Crea retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            k=5,
            search_type="similarity"
        )

        print("Aggiunta primo batch al retriever...")
        retriever.add_documents(first_batch, ids=None)
        print("Primo batch completato!")

        # Processa batch rimanenti
        remaining_docs = all_documents[batch_size:]
        total_batches = (len(remaining_docs) + batch_size - 1) // batch_size

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(remaining_docs))
            batch = remaining_docs[start_idx:end_idx]

            print(f"Batch {i+2}/{total_batches+1}: processamento {len(batch)} documenti...")
            retriever.add_documents(batch, ids=None)
            print(f"Batch {i+2}/{total_batches+1} completato!")

        # Salvataggio
        print("FASE 5: Salvataggio vector store...")
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(os.path.join(vector_store_path, "faiss_index"))

        # Salva docstore (IMPORTANTE per ParentDocumentRetriever)
        docstore_path = os.path.join(vector_store_path, "docstore.pkl")
        with open(docstore_path, 'wb') as f:
            pickle.dump(store, f)

        # Salva metadata
        metadata = {
            "total_documents": len(all_documents),
            "text_documents": len(text_docs),
            "multimodal_documents": len(multimodal_docs),
            "embedding_provider": "huggingface",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
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
    success = main()
    sys.exit(0 if success else 1)