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
        # Chiamiamo create_chatbot con un retriever standard per assicurarci che la cache venga creata.
        # La funzione si occuperà di leggere i PDF, creare l'indice e salvarlo.
        chatbot_instance = create_chatbot(retriever_type="standard")
        
        if chatbot_instance and os.path.exists(VECTOR_STORE_PATH):
            print("\n--- ✅ Processo di creazione della cache completato con successo! ---")
            print(f"L'indice FAISS e i file di cache sono stati salvati in: '{VECTOR_STORE_PATH}'")
            print("Ora puoi eseguire 'git add', 'git commit' e 'git push' per includere la cache nel tuo repository.")
        else:
            print("\n--- ❌ Errore: la creazione della cache non sembra essere andata a buon fine. ---")
            print("Controlla i log sopra per eventuali errori durante l'inizializzazione del chatbot.")

    except Exception as e:
        print(f"\n--- ❌ ERRORE CRITICO durante la creazione della cache: {e} ---")
        print("Il processo è stato interrotto.")

if __name__ == "__main__":
    build_cache()
