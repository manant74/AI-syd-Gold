#!/usr/bin/env python3
"""
Script minimale per costruire rapidamente il vector store.
"""

import os
import sys
import shutil
import pickle
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configurazione diretta
os.environ["EMBEDDING_PROVIDER"] = "huggingface"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("COSTRUZIONE VECTOR STORE VELOCE")
    print("=" * 40)

    # Rimuovi vector store esistente
    vector_store_path = "vector_store_cache"
    if os.path.exists(vector_store_path):
        print("Rimozione vector store esistente...")
        shutil.rmtree(vector_store_path)

    # Carica solo 50 documenti multimodali per test veloce
    cache_dir = Path("cache/multimodal")
    if not cache_dir.exists():
        print("ERRORE: Cache multimodale non trovata!")
        return False

    print("Caricamento documenti multimodali (sample)...")
    pickle_files = list(cache_dir.glob("*.pickle"))[:50]  # Solo primi 50
    documents = []

    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                result = pickle.load(f)
                if hasattr(result, 'text') and result.text.strip():
                    doc = Document(
                        page_content=f"[IMMAGINE-{result.image_type.upper()}] {result.text}",
                        metadata={
                            'source': result.metadata.get('source_pdf', 'unknown'),
                            'page': result.metadata.get('page_num', 0),
                            'content_type': 'multimodal_image'
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"WARN: Errore {pickle_file.name}: {e}")

    print(f"Caricati {len(documents)} documenti multimodali")

    if not documents:
        print("ERRORE: Nessun documento trovato!")
        return False

    # Embeddings semplici
    print("Inizializzazione embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Costruzione vector store diretta (senza ParentDocumentRetriever)
    print("Costruzione vector store...")
    try:
        vector_store = FAISS.from_documents(documents, embeddings)

        # Salvataggio
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(os.path.join(vector_store_path, "faiss_index"))

        # Metadata semplice
        metadata = {
            "total_documents": len(documents),
            "embedding_provider": "huggingface",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }

        import json
        with open(os.path.join(vector_store_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print("OK: Vector store costruito con successo!")
        print(f"Documenti: {len(documents)}")
        return True

    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)