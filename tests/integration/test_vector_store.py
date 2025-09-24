#!/usr/bin/env python3
"""
Test rapido del vector store multimodale.
"""

import os
import sys
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

os.environ["EMBEDDING_PROVIDER"] = "huggingface"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

from app import initialize_embeddings, create_chatbot
from config.settings import AppConfig

def test_vector_store():
    """Test del vector store multimodale."""

    print("TEST VECTOR STORE MULTIMODALE")
    print("=" * 40)

    try:
        # Carica configurazione
        config = AppConfig.from_env()
        print("Configurazione caricata")

        # Inizializza embeddings
        print("Inizializzazione embeddings...")
        embeddings = initialize_embeddings()
        print("Embeddings inizializzati")

        # Crea chatbot
        print("Caricamento chatbot...")
        chatbot = create_chatbot("standard")
        print("Chatbot caricato")

        # Test query semplice
        test_query = "Che tipo di cuscinetti sono descritti nei documenti?"
        print(f"\nQuery di test: {test_query}")
        print("Elaborazione...")

        result = chatbot.invoke({"query": test_query})

        print("\nRISULTATO:")
        print("-" * 20)
        print(result["result"])

        print(f"\nSOURCES ({len(result.get('source_documents', []))} documenti):")
        print("-" * 20)
        for i, doc in enumerate(result.get('source_documents', [])[:3], 1):
            content_type = doc.metadata.get('content_type', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'N/A')

            print(f"{i}. Tipo: {content_type} | Fonte: {source} | Pagina: {page}")

            # Mostra preview contenuto
            content_preview = doc.page_content[:100]
            if len(doc.page_content) > 100:
                content_preview += "..."
            print(f"   Contenuto: {content_preview}")
            print()

        return True

    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_store()
    print("\n" + "=" * 40)
    if success:
        print("OK: Vector store multimodale funziona correttamente!")
    else:
        print("ERRORE: Test fallito")

    sys.exit(0 if success else 1)