#!/usr/bin/env python3
"""
Test script per il sistema multimodale AI-syd-Gold.
Testa l'estrazione di contenuto da immagini e diagrammi nei PDF.
"""

import os
import sys
import logging
from pathlib import Path

# Aggiungi il percorso principale al PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import AppConfig
from extensions.multimodal import MultimodalDocumentProcessor

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_multimodal_processing():
    """Test completo del processamento multimodale."""

    print("TEST SISTEMA MULTIMODALE AI-syd-Gold")
    print("=" * 50)

    # Carica configurazione
    config = AppConfig.from_env()
    print(f"Directory PDF: {config.pdf_directory}")
    print(f"Multimodale abilitato: {config.enable_multimodal}")

    if not config.enable_multimodal:
        print("ERRORE: Processamento multimodale disabilitato in configurazione")
        return False

    # Controlla directory PDF
    if not os.path.exists(config.pdf_directory):
        print(f"ERRORE: Directory PDF non trovata: {config.pdf_directory}")
        return False

    # Trova file PDF
    pdf_files = [f for f in os.listdir(config.pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"ERRORE: Nessun file PDF trovato in {config.pdf_directory}")
        return False

    print(f"Trovati {len(pdf_files)} file PDF:")
    for pdf_file in pdf_files:
        file_path = os.path.join(config.pdf_directory, pdf_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   - {pdf_file} ({file_size:.1f} MB)")

    # Inizializza processore multimodale
    print("\nInizializzazione processore multimodale...")
    try:
        processor = MultimodalDocumentProcessor(config)
        print("OK: Processore multimodale inizializzato")
    except Exception as e:
        print(f"ERRORE inizializzazione: {e}")
        return False

    # Test su primo PDF
    test_pdf = pdf_files[0]
    test_path = os.path.join(config.pdf_directory, test_pdf)

    print(f"\nTest processamento: {test_pdf}")
    print("-" * 30)

    try:
        # Processa PDF
        documents, stats = processor.process_pdf_document(test_path)

        # Mostra statistiche
        print(f"STATISTICHE PROCESSAMENTO:")
        print(f"   - Pagine totali: {stats.total_pages}")
        print(f"   - Immagini trovate: {stats.images_found}")
        print(f"   - Immagini processate: {stats.images_processed}")
        print(f"   - Caratteri estratti: {stats.text_extracted_chars}")
        print(f"   - Errori: {stats.processing_errors}")

        print(f"\nDOCUMENTI ESTRATTI: {len(documents)}")

        # Mostra contenuto estratto
        for i, doc in enumerate(documents[:3]):  # Primi 3 documenti
            print(f"\nDocumento {i+1}:")
            print(f"   Tipo: {doc.metadata.get('image_type', 'unknown')}")
            print(f"   Pagina: {doc.metadata.get('page', 'unknown')}")
            print(f"   Confidenza: {doc.metadata.get('confidence', 0):.1f}%")

            # Mostra preview contenuto
            content_preview = doc.page_content[:200]
            if len(doc.page_content) > 200:
                content_preview += "..."
            print(f"   Contenuto: {content_preview}")

        # Test specifici per documenti tecnici
        print(f"\nANALISI CONTENUTO TECNICO:")
        technical_terms = ['bearing', 'cuscinetto', 'diameter', 'diametro', 'rpm', 'mm', 'tolerance', 'tolleranza']
        found_terms = set()

        for doc in documents:
            content_lower = doc.page_content.lower()
            for term in technical_terms:
                if term in content_lower:
                    found_terms.add(term)

        print(f"   Termini tecnici trovati: {list(found_terms)}")

        # Analisi tipi di contenuto
        content_types = {}
        for doc in documents:
            content_type = doc.metadata.get('image_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1

        print(f"   Tipi di contenuto estratti: {dict(content_types)}")

        # Mostra statistiche processore
        proc_stats = processor.get_processing_statistics()
        print(f"\nSTATISTICHE CACHE:")
        print(f"   - Entries in cache: {proc_stats['cache_entries']}")
        print(f"   - Dimensione cache: {proc_stats['cache_size_mb']:.2f} MB")

        success = len(documents) > 0
        if success:
            print("\nOK: TEST MULTIMODALE COMPLETATO CON SUCCESSO")
        else:
            print("\nWARNING: Test completato ma nessun contenuto estratto")

        return success

    except Exception as e:
        print(f"ERRORE durante processamento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Funzione principale del test."""

    try:
        success = test_multimodal_processing()

        print("\n" + "=" * 50)
        if success:
            print("Test multimodale COMPLETATO")
            print("OK: Il sistema puo estrarre contenuto da immagini e diagrammi")
        else:
            print("Test multimodale FALLITO")
            print("HELP: Controlla le dipendenze e la configurazione")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTest interrotto dall'utente")
        return 1
    except Exception as e:
        print(f"\nERRORE generale: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)