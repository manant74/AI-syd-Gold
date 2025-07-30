# /progetto_chatbot_pdf/app.py

import os
import json
import hashlib
import pickle
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import delle classi necessarie da LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import HypotheticalDocumentEmbedder
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
# Import per la strategia di recupero avanzata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente dal file .env (la nostra chiave API Google)
load_dotenv()

# --- CONFIGURAZIONE PER DEPLOY SEMPLICE (es. Streamlit Cloud) ---
# I PDF devono essere in una cartella 'pdfs' nel repository.
# La cache verrà creata in una cartella 'vector_store_cache' (sarà effimera).
PDF_DIRECTORY_PATH = os.getenv("PDF_DIRECTORY_PATH", "pdfs")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store_cache")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")

# Parametri per il retriever, per un controllo più fine sulla ricerca
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "20"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")  # "similarity" o "mmr"

METADATA_FILE = os.path.join(VECTOR_STORE_PATH, "metadata.json") if VECTOR_STORE_PATH else None

def validate_environment():
    """Valida le variabili d'ambiente e i percorsi."""
    if not PDF_DIRECTORY_PATH:
        raise ValueError("PDF_DIRECTORY_PATH non impostato nel file .env")
    if not VECTOR_STORE_PATH:
        raise ValueError("VECTOR_STORE_PATH non impostato nel file .env")
    if not os.path.isdir(PDF_DIRECTORY_PATH):
        # Per i deploy cloud, la directory potrebbe non esistere al primo avvio
        raise ValueError(f"Directory PDF non trovata: {PDF_DIRECTORY_PATH}")
    logger.info("Validazione ambiente completata con successo")

def _get_pdf_hash(filepath):
    """Calcola l'hash MD5 di un file PDF."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Errore nel calcolo dell'hash per {filepath}: {e}")
        return None

def _get_pdf_metadata(directory):
    """
    Scansiona la directory per i file PDF e restituisce un dizionario 
    con i nomi dei file e i loro hash MD5.
    """
    metadata = {}
    if not os.path.isdir(directory):
        logger.warning(f"Directory non trovata: {directory}")
        return metadata
    
    pdf_count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            file_hash = _get_pdf_hash(filepath)
            if file_hash:
                metadata[filename] = file_hash
                pdf_count += 1
            else:
                logger.warning(f"Impossibile calcolare hash per {filename}")
    
    logger.info(f"Trovati {pdf_count} file PDF nella directory")
    return metadata

def initialize_embeddings():
    """Inizializza e testa l'embedding Google AI."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Test dell'embedding per verificare la connessione
        test_embedding = embeddings.embed_query("test")
        if not test_embedding:
            raise RuntimeError("Test embedding fallito - nessun risultato")
        logger.info("Embeddings Google AI inizializzati con successo")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Errore nell'inizializzazione dell'embedding Google AI: {e}")

def debug_pdf_content(documents):
    """Debug del contenuto dei PDF caricati con analisi dettagliata delle pagine."""
    logger.info("=== DEBUG CONTENUTO PDF ===")
    print("\n=== DEBUG CONTENUTO PDF ===")
    
    if not documents:
        logger.warning("Nessun documento da analizzare")
        print("Nessun documento da analizzare")
        return False
    
    # Raggruppa documenti per file sorgente per analizzare le pagine
    files_analysis = {}
    for doc in documents:
        source = doc.metadata.get('source', 'N/A')
        if source not in files_analysis:
            files_analysis[source] = []
        files_analysis[source].append(doc)
    
    print(f"Analisi di {len(files_analysis)} file PDF:")
    print(f"Totale documenti/pagine caricate: {len(documents)}")
    
    meaningful_files = 0
    for file_path, docs in files_analysis.items():
        filename = os.path.basename(file_path)
        print(f"\n📄 FILE: {filename}")
        print(f"   Pagine caricate: {len(docs)}")
        
        total_chars = sum(len(doc.page_content) for doc in docs)
        total_words = sum(len(doc.page_content.split()) for doc in docs)
        
        print(f"   Caratteri totali: {total_chars:,}")
        print(f"   Parole totali: {total_words:,}")
        
        # Analizza alcune pagine specifiche
        meaningful_pages = 0
        for i, doc in enumerate(docs[:5]):  # Prime 5 pagine
            page_num = doc.metadata.get('page', i)
            page_words = len(doc.page_content.split())
            page_chars = len(doc.page_content)
            
            print(f"   📃 Pagina {page_num + 1}: {page_chars} char, {page_words} parole")
            
            if page_words > 20:
                meaningful_pages += 1
                print(f"      Inizio: '{doc.page_content[:100]}...'")
            else:
                print(f"      ⚠️  Contenuto scarso: '{doc.page_content[:100]}'")
        
        if len(docs) > 5:
            print(f"   ... e altre {len(docs) - 5} pagine")
        
        # Verifica se il file ha contenuto significativo
        if total_words > 100 and meaningful_pages > 0:
            meaningful_files += 1
            print(f"   ✅ File con contenuto significativo")
        else:
            print(f"   ❌ File con contenuto insufficiente")
            logger.warning(f"File {filename} ha contenuto insufficiente: {total_words} parole totali")
    
    print(f"\n📊 RIEPILOGO:")
    print(f"   File con contenuto significativo: {meaningful_files}/{len(files_analysis)}")
    print(f"   Totale pagine elaborate: {len(documents)}")
    print("========================\n")
    
    if meaningful_files == 0:
        logger.error("NESSUN FILE CON CONTENUTO SIGNIFICATIVO TROVATO!")
        print("❌ ERRORE: Nessun file sembra avere contenuto testuale significativo!")
        print("   Possibili cause:")
        print("   - PDF protetti o crittografati")
        print("   - PDF scansionati (solo immagini)")
        print("   - PDF corrotti")
        print("   - Solo prime pagine con header/copyright")
        print("   - Problemi con il parser PDF")
        return False
    
    return True

def load_and_validate_documents():
    """Carica i documenti PDF e verifica che non siano vuoti, con test di loader alternativi."""
    try:
        # Prova il loader principale
        logger.info("Tentativo di caricamento documenti con PyPDFDirectoryLoader...")
        loader = PyPDFDirectoryLoader(PDF_DIRECTORY_PATH, glob="**/*.pdf")
        documents = loader.load()
        
        logger.info(f"PyPDFDirectoryLoader ha caricato {len(documents)} documenti")
        
        # Se non abbiamo documenti o sono insufficienti, proviamo loader alternativi
        if not documents or len(documents) < 5:
            logger.warning("Pochi documenti caricati, tentativo con loader alternativi...")
            
            try:
                # Prova con UnstructuredPDFLoader per singoli file
                from langchain_community.document_loaders import UnstructuredPDFLoader
                alternative_docs = []
                
                for filename in os.listdir(PDF_DIRECTORY_PATH):
                    if filename.lower().endswith('.pdf'):
                        file_path = os.path.join(PDF_DIRECTORY_PATH, filename)
                        try:
                            alt_loader = UnstructuredPDFLoader(file_path)
                            file_docs = alt_loader.load()
                            alternative_docs.extend(file_docs)
                            logger.info(f"UnstructuredPDFLoader caricato {len(file_docs)} docs da {filename}")
                        except Exception as e:
                            logger.warning(f"UnstructuredPDFLoader fallito per {filename}: {e}")
                
                if len(alternative_docs) > len(documents):
                    logger.info(f"UnstructuredPDFLoader più efficace: {len(alternative_docs)} vs {len(documents)}")
                    documents = alternative_docs
                    
            except ImportError:
                logger.warning("UnstructuredPDFLoader non disponibile, installa con: pip install unstructured")
            except Exception as e:
                logger.warning(f"Loader alternativo fallito: {e}")
        
        if not documents:
            logger.warning(f"Nessun documento PDF trovato in '{PDF_DIRECTORY_PATH}'.")
            print("Assicurati che la directory contenga file PDF validi e leggibili.")
            return None
        
        logger.info(f"Totale documenti caricati: {len(documents)}")
        
        # Debug approfondito del contenuto
        if not debug_pdf_content(documents):
            print("\n🔧 SUGGERIMENTI PER RISOLVERE IL PROBLEMA:")
            print("1. Verifica che i PDF non siano protetti da password")
            print("2. Verifica che i PDF non siano solo immagini scansionate")
            print("3. Prova a convertire i PDF in formato testo standard")
            print("4. Usa strumenti come 'pdfinfo' o 'pdftotext' per testare i PDF")
            print("5. Installa parser aggiuntivi: pip install unstructured pdfplumber pymupdf")
            
            # Mostra informazioni sui file
            print(f"\n📁 FILE NELLA DIRECTORY {PDF_DIRECTORY_PATH}:")
            for filename in os.listdir(PDF_DIRECTORY_PATH):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(PDF_DIRECTORY_PATH, filename)
                    file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                    print(f"   - {filename} ({file_size:.1f} MB)")
            
            # Chiedi all'utente se vuole procedere comunque
            logger.error("Nessun contenuto significativo trovato nei PDF. Impossibile procedere.")
            raise ValueError("Nessun contenuto significativo trovato nei PDF. Controlla i file e riprova.")
        
        return documents
        
    except Exception as e:
        logger.error(f"Errore nel caricamento dei documenti: {e}")
        print(f"Errore nel caricamento: {e}")
        
        # Suggerisci alternative
        print("\n🔧 ALTERNATIVE DA PROVARE:")
        print("1. Installare librerie aggiuntive:")
        print("   pip install pypdf2 pdfplumber pymupdf unstructured")
        print("2. Convertire i PDF manualmente in formato testo")
        print("3. Verificare i permessi di accesso ai file")
        print("4. Testare manualmente con: pdftotext file.pdf")
        
        raise

def debug_text_chunks(text_chunks):
    """Debug dei chunk di testo creati."""
    logger.info("=== DEBUG CHUNKS DI TESTO ===")
    print("\n=== DEBUG CHUNKS DI TESTO ===")
    
    if not text_chunks:
        logger.error("Nessun chunk di testo creato!")
        print("❌ Nessun chunk di testo creato!")
        return False
    
    meaningful_chunks = 0
    for i, chunk in enumerate(text_chunks[:5]):  # Analizza primi 5 chunks
        words_count = len(chunk.page_content.split())
        
        print(f"\nChunk {i+1}:")
        print(f"  Lunghezza: {len(chunk.page_content)} caratteri")
        print(f"  Parole: {words_count}")
        print(f"  Contenuto: '{chunk.page_content[:200]}...'")
        
        if words_count > 10:
            meaningful_chunks += 1
    
    logger.info(f"Chunks significativi: {meaningful_chunks}/{len(text_chunks[:5])}")
    print(f"\nChunks con contenuto significativo: {meaningful_chunks}/{min(5, len(text_chunks))}")
    print(f"Totale chunks: {len(text_chunks)}")
    print("=======================\n")
    
    return meaningful_chunks > 0

def create_chatbot(retriever_type="standard"):
    """
    Funzione principale per creare e configurare il chatbot.
    :param retriever_type: "standard", "hyde", o "multi-query"
    """
    # Validazione ambiente
    # Per i deploy semplici, creiamo le directory se non esistono
    if not os.path.exists(PDF_DIRECTORY_PATH):
        os.makedirs(PDF_DIRECTORY_PATH)
        logger.warning(f"Directory PDF '{PDF_DIRECTORY_PATH}' creata, ma è vuota. Assicurati di aggiungere i PDF al repository.")
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    validate_environment()
    
    final_retriever = None
    base_embeddings = initialize_embeddings()

    if retriever_type == "ensemble":
        logger.info("Modalità Ensemble Retriever selezionata. Verrà creato un indice specifico per questa sessione (senza cache).")
        
        documents = load_and_validate_documents()
        if not documents: return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Documenti suddivisi in {len(chunks)} chunks per la modalità Ensemble.")

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = RETRIEVER_K

        vectorstore = FAISS.from_documents(chunks, base_embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

        final_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
    else:
        # Logica esistente per le altre modalità con ParentDocumentRetriever e cache
        embeddings_for_query = base_embeddings
        if retriever_type == "hyde":
            logger.info("Strategia di recupero HyDE selezionata.")
            llm_for_hyde = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)
            hyde_prompt_template = """Sei un assistente utile. Il tuo compito è generare un breve paragrafo che risponda alla domanda dell'utente.
Questo paragrafo verrà utilizzato per la ricerca semantica per trovare i documenti più pertinenti.
Domanda: {question}
Paragrafo di risposta ipotetico:"""
            custom_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)
            embeddings_for_query = HypotheticalDocumentEmbedder.from_llm(
                llm=llm_for_hyde,
                base_embeddings=base_embeddings,
                custom_prompt=custom_prompt
            )

        base_retriever = None
        faiss_index_path = os.path.join(VECTOR_STORE_PATH, "faiss_index")
        faiss_core_index_file = os.path.join(faiss_index_path, "index.faiss")
        docstore_path = os.path.join(VECTOR_STORE_PATH, "docstore.pkl")

        is_cache_valid = False
        # Controllo di robustezza: verifica che i file di cache esistano e non siano vuoti.
        # Questo previene errori in ambienti cloud dove i file potrebbero essere creati ma vuoti.
        if (os.path.exists(METADATA_FILE) and os.path.getsize(METADATA_FILE) > 0 and
            os.path.exists(faiss_core_index_file) and os.path.getsize(faiss_core_index_file) > 0 and
            os.path.exists(docstore_path) and os.path.getsize(docstore_path) > 0):
            try:
                with open(METADATA_FILE, 'r') as f: saved_metadata = json.load(f)
                current_metadata = _get_pdf_metadata(PDF_DIRECTORY_PATH)
                
                # Confronto dettagliato per un logging più chiaro
                saved_files = set(saved_metadata.keys())
                current_files = set(current_metadata.keys())
                added_files = current_files - saved_files
                removed_files = saved_files - current_files
                modified_files = {f for f in saved_files.intersection(current_files) if saved_metadata[f] != current_metadata[f]}

                if not added_files and not removed_files and not modified_files:
                    if current_metadata:
                        logger.info("I documenti non sono cambiati. La cache è valida.")
                        is_cache_valid = True
                else:
                    logger.info("Rilevate modifiche nei documenti. L'indice verrà ricreato.")
                    if added_files: logger.info(f"  File aggiunti: {list(added_files)}")
                    if removed_files: logger.info(f"  File rimossi: {list(removed_files)}")
                    if modified_files: logger.info(f"  File modificati: {list(modified_files)}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Errore nella lettura dei metadati: {e}. L'indice verrà ricreato.")

        if is_cache_valid:
            logger.info("L'indice è aggiornato. Caricamento del retriever dalla cache...")
            try:
                vectorstore = FAISS.load_local(faiss_index_path, embeddings_for_query, allow_dangerous_deserialization=True)
                with open(docstore_path, "rb") as f: store = pickle.load(f)
                base_retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore, docstore=store,
                    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100),
                    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200),
                )
                logger.info("Retriever di base caricato con successo dalla cache.")
            except Exception as e:
                logger.error(f"Errore nel caricamento dalla cache: {e}. L'indice verrà ricreato.")
                base_retriever = None
        
        if base_retriever is None:
            logger.info("Creazione di un nuovo indice con ParentDocumentRetriever...")
            documents = load_and_validate_documents()
            if not documents: return None

            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
            vectorstore = FAISS.from_texts(["_"], embedding=base_embeddings)
            vectorstore.delete(list(vectorstore.index_to_docstore_id.values()))
            store = InMemoryStore()
            base_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore, docstore=store,
                child_splitter=child_splitter, parent_splitter=parent_splitter,
            )
            
             # --- NUOVA LOGICA DI BATCHING E RETRY ---
            # Definiamo una funzione interna con retry per aggiungere documenti in modo robusto
            @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
            def add_documents_with_retry(docs):
                """Aggiunge un batch di documenti al retriever, con tentativi automatici."""
                base_retriever.add_documents(docs, ids=None)

            batch_size = 50  # Numero di documenti da processare in ogni batch
            total_docs = len(documents)
            total_batches = (total_docs // batch_size) + (1 if total_docs % batch_size > 0 else 0)
            
            logger.info(f"Inizio aggiunta di {total_docs} documenti in {total_batches} batch...")

            for i in range(0, total_docs, batch_size):
                batch_docs = documents[i:i + batch_size]
                current_batch_num = (i // batch_size) + 1
                logger.info(f"Elaborazione batch {current_batch_num}/{total_batches} (documenti da {i+1} a {min(i+batch_size, total_docs)})...")
                try:
                    add_documents_with_retry(batch_docs)
                except Exception as e:
                    logger.error(f"Batch {current_batch_num} fallito dopo 5 tentativi: {e}")
                    raise  # Interrompe il processo se un batch fallisce definitivamente

            logger.info("Tutti i batch sono stati elaborati con successo. ")
            # --- FINE NUOVA LOGICA ---

            logger.info("Salvataggio del nuovo indice su disco...")
            base_retriever.vectorstore.save_local(faiss_index_path)
            with open(docstore_path, "wb") as f: pickle.dump(base_retriever.docstore, f)
            current_metadata = _get_pdf_metadata(PDF_DIRECTORY_PATH)
            with open(METADATA_FILE, 'w') as f: json.dump(current_metadata, f, indent=2)
            logger.info(f"Nuovo indice e metadati salvati in '{VECTOR_STORE_PATH}'.")

        if retriever_type == "multi-query":
            logger.info("Applicazione del wrapper Multi-Query Retriever.")
            llm_for_mq = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)
            final_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm_for_mq)
        else:
            final_retriever = base_retriever

    # 4. Creazione della catena di Retrieval-Augmented Generation (RAG)
    try:
        logger.info(f"Utilizzo del modello LLM: {LLM_MODEL_NAME}")
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.7)

        # Creiamo un prompt personalizzato per guidare il comportamento del modello
        prompt_template = """Sei AI-syd-Gold, un assistente tecnico esperto in ingegneria dei materiali e meccanica, specializzato in cuscinetti.
Il tuo obiettivo è assistere un ingegnere meccanico fornendo risposte tecniche precise, basate **esclusivamente** sul contesto fornito.

**REGOLE FONDAMENTALI:**
1.  **BASATI SOLO SUL CONTESTO**: La tua unica fonte di conoscenza è il testo fornito di seguito. Non usare mai la tua conoscenza pregressa.
2.  **LINGUAGGIO E TONO**: Usa un tono formale, tecnico e professionale. Utilizza la terminologia ingegneristica presente nel contesto. Sii fattuale e non esprimere opinioni.

**ISTRUZIONI PER LA FORMATTAZIONE DELLA RISPOSTA:**
Organizza le informazioni estratte per la massima chiarezza, usando la seguente gerarchia quando appropriato:
- **Tabelle**: Per dati numerici, confronti e specifiche tecniche.
- **Elenchi puntati/numerati**: Per caratteristiche, procedure, vantaggi/svantaggi.
- **Paragrafi**: Per spiegazioni concettuali, mantenendoli brevi e focalizzati.

---
**CONTESTO FORNITO:**
{context}
---
**DOMANDA:**
{question}

**RISPOSTA TECNICA:**
"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Creiamo la catena che combina il recupero di informazioni (retriever) e il LLM
        # Ora usiamo il nostro nuovo ParentDocumentRetriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" è il metodo più semplice: prende i chunk e li "infila" tutti nel prompt
            retriever=final_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("Chatbot pronto per ricevere domande.")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Errore nella creazione della catena QA: {e}")
        raise

def main():
    """
    Funzione che gestisce l'interazione con l'utente.
    """
    try:
        qa_chain = create_chatbot()
    except Exception as e:
        logger.error(f"Errore critico durante l'inizializzazione: {e}")
        print(f"Impossibile inizializzare il chatbot: {e}")
        print("Verifica la configurazione del file .env e la connessione API.")
        return

    if qa_chain is None:
        print("Inizializzazione fallita. Controlla i log per maggiori dettagli.")
        return

    print("\n=== AI-SYD-GOLD PRONTO ===")
    print("Modalità di recupero: ParentDocumentRetriever")
    print("Questa modalità migliora il contesto fornendo documenti più completi al modello.")
    print("Digita le tue domande sui documenti PDF caricati.")
    print("Comandi disponibili: 'esci', 'quit', 'exit' per terminare")
    print("Comando speciale: 'debug' per analisi dettagliata del retriever\n")

    # Loop infinito per permettere all'utente di fare domande
    while True:
        try:
            user_question = input("\nFai la tua domanda: ").strip()
            
            if user_question.lower() in ["esci", "quit", "exit"]:
                print("Arrivederci!")
                break
            
            if not user_question:
                print("Per favore, inserisci una domanda.")
                continue

            # Comando debug speciale
            if user_question.lower() == "debug":
                print("\n=== ANALISI DETTAGLIATA RETRIEVER ===")
                try:
                    retriever = qa_chain.retriever
                    # Test con varie query
                    test_queries = ["bearing", "cuscinetto", "lubrication", "steel", "material"]
                    for query in test_queries:
                        docs = retriever.invoke(query)
                        print(f"\nQuery '{query}': {len(docs)} documenti")
                        for i, doc in enumerate(docs[:2]):
                            print(f"  Doc {i+1}: {len(doc.page_content)} chars - '{doc.page_content[:100]}...'")
                except Exception as e:
                    print(f"Errore nel debug: {e}")
                continue

            # --- PASSO DI DEBUG: Controlla cosa recupera il retriever ---
            try:
                retriever = qa_chain.retriever
                retrieved_docs = retriever.invoke(user_question)
                logger.info(f"Recuperati {len(retrieved_docs)} documenti per la domanda")
                
                print("\n--- DEBUG: Documenti recuperati dal retriever ---")
                if not retrieved_docs:
                    print("ATTENZIONE: Il retriever non ha restituito alcun documento.")
                    logger.warning("Nessun documento recuperato per la domanda")
                else:
                    for i, doc in enumerate(retrieved_docs):
                        source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                        print(f"  --- Documento {i+1} (da {source_file}) ---")
                        print(f"  Lunghezza: {len(doc.page_content)} caratteri")
                        print(f"  Parole: {len(doc.page_content.split())} parole")
                        print(f"  Contenuto: {doc.page_content[:300]}...")
                        
                        # Verifica se il contenuto è significativo
                        if len(doc.page_content.split()) < 10:
                            print(f"  ⚠️  ATTENZIONE: Contenuto molto scarso!")
                        
                print("--------------------------------------------------\n")
            except Exception as e:
                logger.error(f"Errore durante il recupero manuale per il debug: {e}")
                print(f"Errore nel debug del retriever: {e}")

            # Eseguiamo la catena con la domanda dell'utente
            try:
                response = qa_chain.invoke({"query": user_question})
                
                # Stampiamo la risposta
                print("\n--- Risposta ---")
                print(response["result"])

                # Opzionale: stampare i documenti sorgente usati per la risposta
                print("\n--- Fonti Utilizzate ---")
                if response["source_documents"]:
                    for source in response["source_documents"]:
                        source_file = os.path.basename(source.metadata.get('source', 'File sconosciuto'))
                        print(f"- File: {source_file}")
                        # print(f"  Contenuto: {source.page_content[:200]}...") # Decommenta per vedere un'anteprima del chunk
                else:
                    print("- Nessuna fonte specifica identificata")
                print("--------------------")
                
            except Exception as e:
                logger.error(f"Errore durante l'elaborazione della domanda: {e}")
                print(f"Errore nell'elaborazione della domanda: {e}")
                print("Riprova con una domanda diversa.")
                
        except KeyboardInterrupt:
            print("\n\nInterruzione da tastiera. Arrivederci!")
            break
        except Exception as e:
            logger.error(f"Errore inaspettato nel loop principale: {e}")
            print(f"Errore inaspettato: {e}")
            print("Il programma continuerà, riprova.")


if __name__ == "__main__":
    main()