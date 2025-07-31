# /progetto_chatbot_pdf/build_cache.py

import os
import shutil
from app import create_chatbot, VECTOR_STORE_PATH

def build_cache():
    """
    Esegue il processo di creazione della cache e la salva nella directory specificata.
    Se la cache esiste già, la elimina per forzare una rigenerazione completa.
    """
    print("--- Inizio del processo di creazione della cache ---")

    # Opzionale ma consigliato: pulisce la cache precedente per una rigenerazione pulita
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Trovata una cache esistente in '{VECTOR_STORE_PATH}'. Verrà rimossa per garantirne la coerenza.")
        try:
            shutil.rmtree(VECTOR_STORE_PATH)
            print("Cache precedente rimossa con successo.")
        except OSError as e:
            print(f"Errore durante la rimozione della cache precedente: {e}")
            return

    print("Invocazione della funzione create_chatbot() per generare il nuovo indice...")
    
    try:
        # 1. Creiamo la cache per le strategie basate su ParentDocumentRetriever
        print("\n--- 1. Creazione cache per strategie 'standard', 'hyde', 'multi-query' ---")
        chatbot_instance_standard = create_chatbot(retriever_type="standard")
        
        # 2. Creiamo la cache per la strategia 'ensemble'
        print("\n--- 2. Creazione cache per strategia 'ensemble' ---")
        chatbot_instance_ensemble = create_chatbot(retriever_type="ensemble")

        if chatbot_instance_standard and chatbot_instance_ensemble and os.path.exists(VECTOR_STORE_PATH):
            print("\n--- ✅ Processo di creazione della cache completato con successo! ---")
            print(f"Le cache per tutte le strategie sono state salvate in: '{VECTOR_STORE_PATH}'")
            print("Ora puoi eseguire 'git add .', 'git commit' e 'git push' per includere la cache nel tuo repository.")
        else:
            print("\n--- ❌ Errore: la creazione della cache non sembra essere andata a buon fine. ---")
            print("Controlla i log sopra per eventuali errori durante l'inizializzazione del chatbot.")

    except Exception as e:
        print(f"\n--- ❌ ERRORE CRITICO durante la creazione della cache: {e} ---")
        print("Il processo è stato interrotto.")

if __name__ == "__main__":
    build_cache()
