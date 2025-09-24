#!/usr/bin/env python3
"""
Script semplificato per costruire rapidamente il vector store.
"""

import os
import sys
import shutil
import pickle
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configurazione diretta (bypassando AppConfig)
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
        print("❌ Cache multimodale non trovata!")
        return []

    pickle_files = list(cache_dir.glob("*.pickle"))
    print(f"Trovati {len(pickle_files)} file cache multimodali")

    for i, pickle_file in enumerate(pickle_files, 1):
        if i % 100 == 0:
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
    """Carica documenti testuali dai PDF."""
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    print("Caricamento documenti testuali...")
    loader = PyPDFDirectoryLoader("pdfs", glob="**/*.pdf")
    documents = loader.load()
    print(f"OK: Caricati {len(documents)} documenti testuali")
    return documents

def main():
    print("COSTRUZIONE VECTOR STORE SEMPLIFICATA")
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

    # Inizializza embeddings (senza ottimizzazioni per evitare problemi)
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

    # Costruzione vector store a batch
    print("FASE 4: Costruzione vector store...")
    batch_size = 100
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    print(f"Processamento in {total_batches} batch di {batch_size} documenti...")

    # Primo batch
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
        k=5,
        search_type="similarity"
    )

    print(f"Aggiunta primo batch al retriever...")
    retriever.add_documents(first_batch, ids=None)
    print(f"Batch 1/{total_batches} completato!")

    # Batch rimanenti
    for i in range(1, total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_documents))
        batch = all_documents[start_idx:end_idx]

        print(f"Batch {i+1}/{total_batches}: processamento {len(batch)} documenti ({start_idx+1}-{end_idx})...")
        retriever.add_documents(batch, ids=None)
        print(f"Batch {i+1}/{total_batches} completato! Totale: {end_idx}/{len(all_documents)}")

    # Salvataggio
    print("FASE 5: Salvataggio vector store...")
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
        "embedding_provider": "huggingface",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }

    metadata_path = os.path.join(vector_store_path, "metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Vector store costruito con successo!")
    print(f"Salvato in: {vector_store_path}")
    print(f"Statistiche:")
    print(f"   - Documenti totali: {len(all_documents)}")
    print(f"   - Documenti testo: {len(text_docs)}")
    print(f"   - Documenti multimodali: {len(multimodal_docs)}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)