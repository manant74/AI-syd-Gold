#!/usr/bin/env python3
"""
Script dedicato per costruire il vector store utilizzando il nuovo approccio ibrido.
Usa MultimodalDocumentProcessor come unica fonte per l'estrazione dei dati.
Supporta checkpoint intermedi per riprendere in caso di crash.
"""

import os
import sys
import shutil
import pickle
import json
import argparse
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import AppConfig
from app import initialize_embeddings
from extensions.multimodal import MultimodalDocumentProcessor

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.stores import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

CHECKPOINT_BATCH_SIZE = 200  # Salva checkpoint ogni N documenti


def get_checkpoint_path(vector_store_path: Path) -> Path:
    return vector_store_path / "checkpoint"


def save_checkpoint(checkpoint_path: Path, vector_store: FAISS, store: InMemoryStore,
                    processed_count: int, all_documents: list):
    """Salva stato intermedio su disco."""
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(checkpoint_path / "faiss_index"))
    with open(checkpoint_path / "docstore.pkl", "wb") as f:
        pickle.dump(store, f)
    with open(checkpoint_path / "progress.json", "w") as f:
        json.dump({"processed_count": processed_count, "total": len(all_documents)}, f)
    print(f"  [CHECKPOINT] Salvato a {processed_count}/{len(all_documents)} documenti.")


def load_checkpoint(checkpoint_path: Path, embeddings):
    """Carica checkpoint esistente. Restituisce (vector_store, store, processed_count) o None."""
    progress_file = checkpoint_path / "progress.json"
    if not progress_file.exists():
        return None

    try:
        with open(progress_file) as f:
            progress = json.load(f)
        vector_store = FAISS.load_local(
            str(checkpoint_path / "faiss_index"), embeddings,
            allow_dangerous_deserialization=True
        )
        with open(checkpoint_path / "docstore.pkl", "rb") as f:
            store = pickle.load(f)
        processed_count = progress["processed_count"]
        print(f"  [CHECKPOINT] Ripreso da {processed_count}/{progress['total']} documenti.")
        return vector_store, store, processed_count
    except Exception as e:
        print(f"  [CHECKPOINT] Errore caricamento checkpoint: {e}. Ripartenza da zero.")
        return None


DOCUMENTS_DUMP = "extracted_documents.pkl"


def save_documents_dump(vector_store_path: Path, documents: list):
    """Salva i documenti estratti per poter saltare la fase 2 in futuro."""
    dump_path = vector_store_path / DOCUMENTS_DUMP
    with open(dump_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"  [DUMP] Documenti salvati in {dump_path}")


def load_documents_dump(vector_store_path: Path):
    """Carica i documenti estratti in precedenza."""
    dump_path = vector_store_path / DOCUMENTS_DUMP
    if not dump_path.exists():
        return None
    with open(dump_path, "rb") as f:
        documents = pickle.load(f)
    print(f"  [DUMP] Caricati {len(documents)} documenti da {dump_path}")
    return documents


def build_hybrid_store(from_phase: int = 1):
    """Costruisce il vector store usando solo il processore ibrido.

    Args:
        from_phase: Fase da cui partire (1=tutto, 2=salta config, 3=salta estrazione PDF).
    """

    print("COSTRUZIONE VECTOR STORE IBRIDO")
    print("=" * 50)

    # --- 1. Configurazione ---
    print("FASE 1: Caricamento configurazione...")
    config = AppConfig.from_env()
    vector_store_path = Path(config.vector_store_directory)
    checkpoint_path = get_checkpoint_path(vector_store_path)
    vector_store_path.mkdir(parents=True, exist_ok=True)

    # --- 2. Estrazione Dati con Logica Ibrida ---
    if from_phase <= 2:
        print("\nFASE 2: Estrazione documenti con MultimodalDocumentProcessor...")
        multimodal_processor = MultimodalDocumentProcessor(config)

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

        save_documents_dump(vector_store_path, all_documents)
        print(f"\nEstrazione completata. Totale documenti estratti: {len(all_documents)}")
    else:
        print("\nFASE 2: SALTATA — caricamento documenti da dump precedente...")
        all_documents = load_documents_dump(vector_store_path)
        if all_documents is None:
            print(f"ERRORE: Nessun dump trovato in {vector_store_path / DOCUMENTS_DUMP}.")
            print("Esegui prima senza --from-phase 3 per generare il dump.")
            return False
        print(f"Totale documenti caricati: {len(all_documents)}")

    # --- 3. Creazione Vector Store con Checkpoint ---
    print("\nFASE 3: Costruzione del nuovo vector store FAISS...")
    try:
        embeddings = initialize_embeddings()

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.parent_chunk_size)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=config.child_chunk_size)

        # Prova a caricare checkpoint esistente
        checkpoint_data = load_checkpoint(checkpoint_path, embeddings)
        if checkpoint_data:
            vector_store, store, start_index = checkpoint_data
        else:
            # Primo batch per inizializzare il vector store
            print(f"Creazione vector store iniziale con il primo batch...")
            first_batch = all_documents[:CHECKPOINT_BATCH_SIZE]
            vector_store = FAISS.from_documents(first_batch, embeddings)
            store = InMemoryStore()
            start_index = CHECKPOINT_BATCH_SIZE
            save_checkpoint(checkpoint_path, vector_store, store, start_index, all_documents)

        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # Processa batch rimanenti
        remaining = all_documents[start_index:]
        total = len(all_documents)
        batch_num = 0

        for i in range(0, len(remaining), CHECKPOINT_BATCH_SIZE):
            batch = remaining[i:i + CHECKPOINT_BATCH_SIZE]
            batch_start = start_index + i
            batch_end = min(batch_start + len(batch), total)
            batch_num += 1

            print(f"Batch {batch_num}: documenti {batch_start + 1}-{batch_end}/{total}...")
            retriever.add_documents(batch, ids=None)
            save_checkpoint(checkpoint_path, vector_store, store, batch_end, all_documents)

        # --- Salvataggio finale ---
        print("\nSalvataggio indice finale...")
        faiss_index_path = vector_store_path / "faiss_index"
        docstore_path = vector_store_path / "docstore.pkl"
        metadata_path = vector_store_path / "metadata.json"

        vector_store.save_local(str(faiss_index_path))
        with open(docstore_path, "wb") as f:
            pickle.dump(store, f)

        metadata = {
            "total_documents": len(all_documents),
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "source": "build_hybrid_store.py"
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Rimuovi checkpoint ora che abbiamo l'indice finale
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            print("Checkpoint intermedi rimossi.")

        print("\nOK: Vector store ibrido costruito con successo!")
        print(f"Salvato in: {vector_store_path}")
        print(f"Statistiche: {len(all_documents)} documenti totali.")
        return True

    except Exception as e:
        import traceback
        print(f"\nERRORE CRITICO durante la costruzione del vector store: {e}")
        traceback.print_exc()
        print("L'ultimo checkpoint è stato preservato — rilancia lo script per riprendere.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build hybrid vector store")
    parser.add_argument(
        "--from-phase", type=int, choices=[1, 2, 3], default=1,
        help="Fase da cui partire: 1=tutto (default), 2=salta config, 3=salta estrazione PDF (usa dump esistente)"
    )
    args = parser.parse_args()
    success = build_hybrid_store(from_phase=args.from_phase)
    sys.exit(0 if success else 1)
