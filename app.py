import os
import json
import hashlib
import pickle
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import configurazione centralizzata
from config import AppConfig
from system_prompt import EXPERT_PROMPT, HYDE_PROMPT, CHAIN_OF_THOUGHT_PROMPT, CHAIN_OF_THOUGHT_SYNTHESIS_PROMPT
from utils import MemoryOptimizer, monitor_memory_usage, process_documents_in_chunks

# Import delle classi necessarie da LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from llm_providers import LLMFactory
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
from langchain_core.retrievers import BaseRetriever
# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente dal file .env (la nostra chiave API Google)
# override=True forza l'uso del file .env anche se ci sono variabili di sistema
load_dotenv(override=True)

# Configurazione centralizzata dell'applicazione
config = AppConfig.from_env()

# Inizializzazione del memory optimizer
memory_optimizer = MemoryOptimizer(config)


def validate_environment():
    """Valida le variabili d'ambiente e i percorsi."""
    # Usa la validazione integrata della configurazione
    config.validate()
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
    """Inizializza e testa l'embedding con il provider configurato e ottimizzazioni memoria."""
    with monitor_memory_usage("initialize_embeddings"):
        try:
            logger.info(f"Inizializzazione embeddings: provider={config.embedding_provider}, model={config.embedding_model}")

            # Crea embeddings usando il factory
            embeddings = LLMFactory.create_embeddings(
                provider=config.embedding_provider,
                model=config.embedding_model,
                config=config
            )

            # Test dell'embedding per verificare la connessione
            test_embedding = embeddings.embed_query("test")
            if not test_embedding:
                raise RuntimeError("Test embedding fallito - nessun risultato")

            # Applica ottimizzazioni memoria agli embeddings
            optimized_embeddings = memory_optimizer.create_optimized_embeddings_func(embeddings)

            logger.info(f"Embeddings {config.embedding_provider} inizializzati con successo")
            return optimized_embeddings
        except Exception as e:
            raise RuntimeError(f"Errore nell'inizializzazione dell'embedding {config.embedding_provider}: {e}")

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
    with monitor_memory_usage("load_and_validate_documents"):
        try:
            # Prova il loader principale
            logger.info("Tentativo di caricamento documenti con PyPDFDirectoryLoader...")
            loader = PyPDFDirectoryLoader(config.pdf_directory, glob="**/*.pdf")
            documents = loader.load()

            logger.info(f"PyPDFDirectoryLoader ha caricato {len(documents)} documenti")
        
            # Se non abbiamo documenti o sono insufficienti, proviamo loader alternativi
            if not documents or len(documents) < 5:
                logger.warning("Pochi documenti caricati, tentativo con loader alternativi...")

                try:
                    # Prova con UnstructuredPDFLoader per singoli file
                    from langchain_community.document_loaders import UnstructuredPDFLoader
                    alternative_docs = []

                    for filename in os.listdir(config.pdf_directory):
                        if filename.lower().endswith('.pdf'):
                            file_path = os.path.join(config.pdf_directory, filename)
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
                logger.warning(f"Nessun documento PDF trovato in '{config.pdf_directory}'.")
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
                print(f"\n📁 FILE NELLA DIRECTORY {config.pdf_directory}:")
                for filename in os.listdir(config.pdf_directory):
                    if filename.lower().endswith('.pdf'):
                        file_path = os.path.join(config.pdf_directory, filename)
                        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                        print(f"   - {filename} ({file_size:.1f} MB)")

                # Chiedi all'utente se vuole procedere comunque
                logger.error("Nessun contenuto significativo trovato nei PDF. Impossibile procedere.")
                raise ValueError("Nessun contenuto significativo trovato nei PDF. Controlla i file e riprova.")

            # Garbage collection dopo il caricamento dei documenti
            import gc
            collected = gc.collect()
            if collected > 0:
                logger.info(f"Garbage collection dopo caricamento documenti: {collected} oggetti rimossi")

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

class ChainOfThoughtRetriever(BaseRetriever):
    """
    Retriever che utilizza Chain-of-Thought reasoning per analizzare la query
    e pianificare una strategia di ricerca ottimale.
    """
    base_retriever: object
    config: object
    llm: object = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, base_retriever, config, **kwargs):
        super().__init__(base_retriever=base_retriever, config=config, **kwargs)

    def _get_llm(self):
        """Lazy initialization dell'LLM per reasoning."""
        if self.llm is None:
            self.llm = LLMFactory.create_llm(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                config=self.config,
                temperature=0.1  # Bassa temperatura per reasoning consistente
            )
        return self.llm

    def get_relevant_documents(self, query):
        """
        Utilizza Chain-of-Thought reasoning per recuperare documenti rilevanti.

        Args:
            query (str): La domanda dell'utente

        Returns:
            List[Document]: Documenti rilevanti recuperati attraverso reasoning
        """
        logger.info(f"Avvio Chain-of-Thought reasoning per query: {query[:100]}...")

        try:
            # STEP 1: Analisi della query con Chain-of-Thought
            llm = self._get_llm()

            reasoning_prompt = PromptTemplate(
                input_variables=["question"],
                template=CHAIN_OF_THOUGHT_PROMPT
            )

            logger.info("Esecuzione analisi Chain-of-Thought...")
            reasoning_response = llm.invoke(reasoning_prompt.format(question=query))

            # STEP 2: Estrai le query di ricerca dal reasoning
            search_queries = self._extract_search_queries(reasoning_response, query)
            logger.info(f"Generate {len(search_queries)} query di ricerca specifiche")

            # STEP 3: Esegui ricerche sequenziali
            all_documents = []
            collected_info = []

            for i, search_query in enumerate(search_queries, 1):
                logger.info(f"Esecuzione ricerca {i}/{len(search_queries)}: {search_query[:80]}...")

                # Ricerca documenti per questa query specifica
                docs = self.base_retriever.get_relevant_documents(search_query)

                # Evita duplicati mantenendo solo documenti nuovi
                new_docs = []
                for doc in docs:
                    doc_content = doc.page_content[:200]  # Prime 200 char per confronto
                    if not any(doc_content in existing.page_content for existing in all_documents):
                        new_docs.append(doc)

                all_documents.extend(new_docs[:2])  # Max 2 docs per query per evitare overload

                # Estrai informazioni chiave per il prossimo step
                if new_docs:
                    key_info = self._extract_key_information(new_docs, search_query)
                    collected_info.append(f"Ricerca {i}: {key_info}")

            logger.info(f"Chain-of-Thought completato: {len(all_documents)} documenti recuperati")

            # Limita il numero totale di documenti per performance
            return all_documents[:self.config.retriever_k] if all_documents else []

        except Exception as e:
            logger.error(f"Errore in Chain-of-Thought reasoning: {e}")
            logger.info("Fallback a ricerca standard...")
            return self.base_retriever.get_relevant_documents(query)

    def _extract_search_queries(self, reasoning_response, original_query):
        """Estrae le query di ricerca dal testo di reasoning."""
        try:
            # Cerca sezioni che contengono query numerate
            lines = reasoning_response.split('\n')
            queries = []

            for line in lines:
                line = line.strip()
                # Cerca pattern come "1. [query]" o "Prima ricerca: [query]"
                if any(pattern in line.lower() for pattern in ['1.', '2.', '3.', '4.', 'prima ricerca', 'seconda ricerca', 'terza ricerca']):
                    # Pulisci la line da numerazione e formattazione
                    clean_query = line
                    for prefix in ['1.', '2.', '3.', '4.', 'Prima ricerca:', 'Seconda ricerca:', 'Terza ricerca:', 'Quarta ricerca:']:
                        clean_query = clean_query.replace(prefix, '').strip()

                    # Rimuovi brackets e altri caratteri di formattazione
                    clean_query = clean_query.replace('[', '').replace(']', '').strip()

                    if clean_query and len(clean_query) > 10:  # Query ragionevole
                        queries.append(clean_query)

            # Se non troviamo query strutturate, fallback a query generiche
            if not queries:
                queries = [
                    original_query,  # Query originale
                    f"specifiche tecniche {original_query}",  # Versione tech specs
                    f"procedure manutenzione {original_query}"  # Versione maintenance
                ]

            return queries[:4]  # Max 4 query per performance

        except Exception as e:
            logger.warning(f"Errore estrazione query: {e}")
            return [original_query]

    def _extract_key_information(self, documents, search_query):
        """Estrae informazioni chiave dai documenti per la query corrente."""
        try:
            if not documents:
                return "Nessuna informazione trovata"

            # Combina contenuto dei documenti (primi 300 caratteri di ciascuno)
            combined_content = "\n".join([doc.page_content[:300] for doc in documents])

            # Estrai punti chiave rilevanti per la query
            key_points = combined_content[:500]  # Limita per performance
            return f"Query '{search_query[:50]}...': {key_points[:200]}..."

        except Exception as e:
            logger.warning(f"Errore estrazione informazioni: {e}")
            return "Informazioni estratte con errori"


def create_chatbot(retriever_type="standard"):
    """
    Funzione principale per creare e configurare il chatbot.
    :param retriever_type: "standard", "hyde", o "multi-query"
    """
    # Validazione ambiente
    # Per i deploy semplici, creiamo le directory se non esistono
    if not os.path.exists(config.pdf_directory):
        os.makedirs(config.pdf_directory)
        logger.warning(f"Directory PDF '{config.pdf_directory}' creata, ma è vuota. Assicurati di aggiungere i PDF al repository.")
    os.makedirs(config.vector_store_directory, exist_ok=True)
    validate_environment()
    
    final_retriever = None
    base_embeddings = None  # Inizializzeremo solo se necessario

    if retriever_type == "ensemble":
        with monitor_memory_usage("ensemble_retriever_creation"):
            logger.info("Modalità Ensemble Retriever selezionata. Verrà creato un indice specifico per questa sessione (senza cache).")

            # Inizializza embeddings solo per ensemble (che non usa cache)
            if base_embeddings is None:
                base_embeddings = initialize_embeddings()

            documents = load_and_validate_documents()
            if not documents: return None

            with monitor_memory_usage("document_splitting"):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap
                )
                chunks = text_splitter.split_documents(documents)
                logger.info(f"Documenti suddivisi in {len(chunks)} chunks per la modalità Ensemble.")

            with monitor_memory_usage("bm25_retriever_creation"):
                bm25_retriever = BM25Retriever.from_documents(chunks)
                bm25_retriever.k = config.retriever_k

            with monitor_memory_usage("faiss_vectorstore_creation"):
                vectorstore = FAISS.from_documents(chunks, base_embeddings)
                faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.retriever_k})

            final_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
    else:
        # Logica esistente per le altre modalità con ParentDocumentRetriever e cache
        base_retriever = None
        faiss_index_path = config.faiss_index_path
        faiss_core_index_file = config.faiss_core_index_file
        docstore_path = config.docstore_path

        is_cache_valid = False
        # Controllo di robustezza: verifica che i file di cache esistano e non siano vuoti.
        # Usa SEMPRE la cache se esiste, senza controllare modifiche ai PDF.
        # La rigenerazione avviene solo manualmente tramite il pulsante nell'interfaccia.
        if (os.path.exists(config.metadata_file_path) and os.path.getsize(config.metadata_file_path) > 0 and
            os.path.exists(faiss_core_index_file) and os.path.getsize(faiss_core_index_file) > 0 and
            os.path.exists(docstore_path) and os.path.getsize(docstore_path) > 0):
            try:
                with open(config.metadata_file_path, 'r') as f: saved_metadata = json.load(f)

                # Usa sempre la cache se esiste, indipendentemente dai cambiamenti ai PDF
                if saved_metadata:
                    logger.info("Cache degli embeddings trovata. Caricamento dalla cache (nessun controllo di modifiche).")
                    is_cache_valid = True
                else:
                    logger.info("Cache esistente ma vuota. L'indice verrà ricreato.")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Errore nella lettura dei metadati: {e}. L'indice verrà ricreato.")

        if is_cache_valid:
            logger.info("L'indice è aggiornato. Caricamento del retriever dalla cache...")
            try:
                # Prova a inizializzare embeddings solo se necessario
                # In caso di errore (quota esaurita), usa un placeholder
                if base_embeddings is None:
                    try:
                        base_embeddings = initialize_embeddings()
                    except Exception as e:
                        logger.warning(f"Impossibile inizializzare embeddings: {e}")
                        logger.info("Tentativo di caricamento cache con embeddings placeholder...")

                        # Crea embeddings placeholder che non fanno chiamate API
                        from langchain.embeddings.base import Embeddings
                        import numpy as np
                        import faiss

                        # Prova a determinare le dimensioni dall'indice esistente
                        try:
                            index = faiss.read_index(config.faiss_core_index_file)
                            embedding_dim = index.d
                            logger.info(f"Dimensioni embeddings rilevate dall'indice: {embedding_dim}")
                        except:
                            embedding_dim = 768  # Default per Google embeddings
                            logger.warning(f"Impossibile rilevare dimensioni, uso default: {embedding_dim}")

                        class PlaceholderEmbeddings(Embeddings):
                            def __init__(self, dim):
                                self.dim = dim

                            def embed_documents(self, texts):
                                return [np.zeros(self.dim).tolist() for _ in texts]

                            def embed_query(self, text):
                                return np.zeros(self.dim).tolist()

                        base_embeddings = PlaceholderEmbeddings(embedding_dim)

                embeddings_for_query = base_embeddings
                if retriever_type == "hyde":
                    logger.info("Strategia di recupero HyDE selezionata.")
                    logger.warning("HyDE non disponibile con embeddings placeholder - uso standard")
                    # Non possiamo usare HyDE con placeholder, fallback a standard

                vectorstore = FAISS.load_local(faiss_index_path, embeddings_for_query, allow_dangerous_deserialization=True)
                with open(docstore_path, "rb") as f: store = pickle.load(f)
                base_retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore, docstore=store,
                    child_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=config.child_chunk_size,
                        chunk_overlap=config.child_chunk_overlap
                    ),
                    parent_splitter=RecursiveCharacterTextSplitter(
                        chunk_size=config.parent_chunk_size,
                        chunk_overlap=config.parent_chunk_overlap
                    ),
                )
                logger.info("Retriever di base caricato con successo dalla cache.")
            except Exception as e:
                logger.warning(f"Errore nel caricamento dalla cache (possibile incompatibilità embeddings): {e}")
                logger.info("Cache incompatibile con il nuovo provider embeddings. Usa il pulsante 'Rigenera embeddings' per ricreare la cache.")
                base_retriever = None
        
        if base_retriever is None:
            # Se la cache non è valida ma non vogliamo rigenerare automaticamente
            if is_cache_valid is False:
                logger.info("Nessuna cache valida trovata. Usa il pulsante 'Rigenera embeddings' per creare gli indici.")
                return None

            logger.info("Creazione di un nuovo indice con ParentDocumentRetriever...")

            # Inizializza embeddings solo se necessario per creare nuovo indice
            if base_embeddings is None:
                base_embeddings = initialize_embeddings()

            embeddings_for_query = base_embeddings
            if retriever_type == "hyde":
                logger.info("Strategia di recupero HyDE selezionata.")
                llm_for_hyde = LLMFactory.create_llm(
                    provider=config.llm_provider,
                    model=config.llm_model,
                    config=config,
                    temperature=0
                )
                custom_prompt = PromptTemplate(input_variables=["question"], template=HYDE_PROMPT)
                embeddings_for_query = HypotheticalDocumentEmbedder.from_llm(
                    llm=llm_for_hyde,
                    base_embeddings=base_embeddings,
                    custom_prompt=custom_prompt
                )

            documents = load_and_validate_documents()
            if not documents: return None

            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.parent_chunk_size,
                chunk_overlap=config.parent_chunk_overlap
            )
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.child_chunk_size,
                chunk_overlap=config.child_chunk_overlap
            )
            vectorstore = FAISS.from_texts(["_"], embedding=base_embeddings)
            vectorstore.delete(list(vectorstore.index_to_docstore_id.values()))
            store = InMemoryStore()
            base_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore, docstore=store,
                child_splitter=child_splitter, parent_splitter=parent_splitter,
            )
            
             # --- LOGICA DI BATCHING OTTIMIZZATA CON MEMORY MANAGEMENT ---
            # Definiamo una funzione interna con retry per aggiungere documenti in modo robusto
            @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
            def add_documents_with_retry(docs):
                """Aggiunge un batch di documenti al retriever, con tentativi automatici."""
                with monitor_memory_usage(f"add_documents_batch_{len(docs)}"):
                    base_retriever.add_documents(docs, ids=None)

            total_docs = len(documents)
            logger.info(f"Inizio processamento ottimizzato di {total_docs} documenti...")

            # Usa il memory optimizer per determinare il batch size ottimale e processare i documenti
            batch_num = 0
            for batch in memory_optimizer.optimize_batch_processing(documents, "document_indexing"):
                batch_num += 1
                logger.info(f"Elaborazione batch ottimizzato {batch_num} con {len(batch)} documenti...")
                try:
                    add_documents_with_retry(batch)
                except Exception as e:
                    logger.error(f"Batch {batch_num} fallito dopo 5 tentativi: {e}")
                    raise  # Interrompe il processo se un batch fallisce definitivamente

            logger.info("Tutti i batch sono stati elaborati con successo con ottimizzazioni memoria.")
            # --- FINE LOGICA OTTIMIZZATA ---

            with monitor_memory_usage("save_index_to_disk"):
                logger.info("Salvataggio del nuovo indice su disco...")
                base_retriever.vectorstore.save_local(faiss_index_path)
                with open(docstore_path, "wb") as f: pickle.dump(base_retriever.docstore, f)
                current_metadata = _get_pdf_metadata(config.pdf_directory)
                with open(config.metadata_file_path, 'w') as f: json.dump(current_metadata, f, indent=2)
                logger.info(f"Nuovo indice e metadati salvati in '{config.vector_store_directory}'.")

                # Log del report memoria finale
                memory_report = memory_optimizer.get_memory_report()
                logger.info(f"Report memoria finale: {memory_report['current_memory_mb']:.1f} MB utilizzati, "
                           f"picco: {memory_report['peak_memory_mb']:.1f} MB")

        if retriever_type == "multi-query":
            logger.info("Applicazione del wrapper Multi-Query Retriever.")
            llm_for_mq = LLMFactory.create_llm(
                provider=config.llm_provider,
                model=config.llm_model,
                config=config,
                temperature=0
            )
            final_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm_for_mq)
        elif retriever_type == "chain-of-thought":
            logger.info("Modalità Chain-of-Thought Reasoning selezionata.")
            final_retriever = ChainOfThoughtRetriever(base_retriever, config)
        else:
            final_retriever = base_retriever

    # 4. Creazione della catena di Retrieval-Augmented Generation (RAG)
    try:
        logger.info(f"Utilizzo del modello LLM: {config.llm_model}")
        llm = LLMFactory.create_llm(
            provider=config.llm_provider,
            model=config.llm_model,
            config=config,
            temperature=0.7
        )

        # Utilizziamo il prompt centralizzato dal modulo system_prompt

        PROMPT = PromptTemplate(
            template=EXPERT_PROMPT, input_variables=["context", "question"]
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

def log_memory_stats_if_enabled():
    """Log dettagliato delle statistiche memoria se il debug è abilitato."""
    if logger.isEnabledFor(logging.INFO):
        memory_report = memory_optimizer.get_memory_report()
        logger.info("=== MEMORY REPORT ===")
        logger.info(f"Memoria corrente: {memory_report['current_memory_mb']:.1f} MB")
        logger.info(f"Memoria baseline: {memory_report['baseline_memory_mb']:.1f} MB")
        logger.info(f"Delta: {memory_report['delta_mb']:+.1f} MB")
        logger.info(f"Picco memoria: {memory_report['peak_memory_mb']:.1f} MB")
        logger.info(f"% processo: {memory_report['process_percent']:.1f}%")
        logger.info(f"% sistema: {memory_report['system_percent']:.1f}%")
        logger.info(f"Memoria disponibile: {memory_report['available_mb']:.1f} MB")

        # Cache stats
        cache_info = memory_report['cache_stats']['embed_query_cache']
        if cache_info['currsize'] > 0:
            hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses']) * 100
            logger.info(f"Cache embeddings: {cache_info['currsize']}/{cache_info['maxsize']} "
                       f"(hit rate: {hit_rate:.1f}%)")
        logger.info("====================")

def cleanup_memory_if_needed():
    """Pulisce la memoria se necessario basandosi sui thresholds."""
    from utils import clear_caches

    memory_info = memory_optimizer.monitor.get_memory_info()

    # Cleanup aggressivo se la memoria è alta
    if memory_info['system_percent'] > 90 or memory_info['percent'] > 80:
        logger.warning(f"Alta pressione memoria rilevata (sistema: {memory_info['system_percent']:.1f}%, "
                      f"processo: {memory_info['percent']:.1f}%). Pulizia in corso...")
        clear_caches()
        import gc
        collected = gc.collect()
        logger.info(f"Cleanup memoria completato. Oggetti rimossi: {collected}")

        # Log memoria dopo cleanup
        new_memory_info = memory_optimizer.monitor.get_memory_info()
        logger.info(f"Memoria dopo cleanup: {new_memory_info['rss_mb']:.1f} MB "
                   f"(processo: {new_memory_info['percent']:.1f}%)")

def main():
    """
    Funzione che gestisce l'interazione con l'utente.
    """
    try:
        with monitor_memory_usage("chatbot_initialization"):
            qa_chain = create_chatbot()
            log_memory_stats_if_enabled()
    except Exception as e:
        logger.error(f"Errore critico durante l'inizializzazione: {e}")
        print(f"Impossibile inizializzare il chatbot: {e}")
        print("Verifica la configurazione del file .env e la connessione API.")
        return

    if qa_chain is None:
        print("Inizializzazione fallita. Controlla i log per maggiori dettagli.")
        return

    print("\n=== BEARX PRONTO ===")
    print("Modalità di recupero: ParentDocumentRetriever")
    print("Questa modalità migliora il contesto fornendo documenti più completi al modello.")
    print("Digita le tue domande sui documenti PDF caricati.")
    print("Comandi disponibili: 'esci', 'quit', 'exit' per terminare")
    print("Comando speciale: 'debug' per analisi dettagliata del retriever")
    print("Comando speciale: 'memory' per statistiche memoria")
    print("Comando speciale: 'llm' per visualizzare configurazione LLM\n")

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

            # Comando llm speciale per cambiare provider
            if user_question.lower() == "llm":
                print("\n=== CONFIGURAZIONE LLM ===")
                from llm_providers import LLMFactory

                available = LLMFactory.get_available_models()
                print("Provider disponibili:")
                for i, (provider, models) in enumerate(available.items(), 1):
                    current = "(ATTUALE)" if provider == config.llm_provider else ""
                    print(f"  {i}. {provider} {current}")
                    if models["llm"]:
                        print(f"     LLM: {', '.join(models['llm'][:3])}{'...' if len(models['llm']) > 3 else ''}")
                    if models["embeddings"]:
                        print(f"     Embeddings: {', '.join(models['embeddings'][:2])}{'...' if len(models['embeddings']) > 2 else ''}")

                print(f"\nAttuale: {config.llm_provider}/{config.llm_model}")
                print(f"Embeddings: {config.embedding_provider}/{config.embedding_model}")
                print("Per cambiare provider, modifica LLM_PROVIDER nel file .env")
                print("===========================")
                continue

            # Comando memory speciale
            if user_question.lower() == "memory":
                print("\n=== STATISTICHE MEMORIA ===")
                memory_report = memory_optimizer.get_memory_report()
                print(f"Memoria corrente: {memory_report['current_memory_mb']:.1f} MB")
                if memory_report['baseline_memory_mb']:
                    print(f"Memoria baseline: {memory_report['baseline_memory_mb']:.1f} MB")
                    print(f"Delta: {memory_report['delta_mb']:+.1f} MB")
                print(f"Picco memoria: {memory_report['peak_memory_mb']:.1f} MB")
                print(f"% processo: {memory_report['process_percent']:.1f}%")
                print(f"% sistema: {memory_report['system_percent']:.1f}%")
                print(f"Memoria disponibile: {memory_report['available_mb']:.1f} MB")

                cache_info = memory_report['cache_stats']['embed_query_cache']
                if cache_info['currsize'] > 0:
                    hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses']) * 100
                    print(f"Cache embeddings: {cache_info['currsize']}/{cache_info['maxsize']} "
                          f"(hit rate: {hit_rate:.1f}%)")
                print("===========================")

                # Cleanup se necessario
                cleanup_memory_if_needed()
                continue

            # Comando llm speciale
            if user_question.lower() == "llm":
                print("\n=== CONFIGURAZIONE LLM ATTUALE ===")
                print(f"Provider LLM: {config.llm_provider}")
                print(f"Modello LLM: {config.llm_model}")
                print(f"Provider Embeddings: {config.embedding_provider}")
                print(f"Modello Embeddings: {config.embedding_model}")

                print("\n=== MODELLI DISPONIBILI ===")
                available_models = LLMFactory.get_available_models()
                for provider, models in available_models.items():
                    if models["llm"]:
                        print(f"\n{provider.upper()} LLM:")
                        for model in models["llm"]:
                            current = " (ATTUALE)" if provider == config.llm_provider and model == config.llm_model else ""
                            print(f"  - {model}{current}")

                print("\nPer cambiare modello, modifica il file .env e riavvia l'applicazione")
                print("================================")
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

            # Eseguiamo la catena con la domanda dell'utente con monitoraggio memoria
            with monitor_memory_usage(f"query_processing"):
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

                    # Cleanup periodico della memoria
                    cleanup_memory_if_needed()

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