import os
import gc
import hashlib
import pickle
import logging
from dotenv import load_dotenv

# Import configurazione centralizzata
from config import AppConfig
from config.system_prompt import CHAIN_OF_THOUGHT_PROMPT, EXPERT_PROMPT, HYDE_PROMPT

# Import delle classi necessarie da LangChain
from config.llm_providers import LLMFactory
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Import per la strategia di recupero avanzata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente dal file .env (la nostra chiave API Google)
# override=True forza l'uso del file .env anche se ci sono variabili di sistema
load_dotenv(override=True)

# Configurazione centralizzata dell'applicazione
config = AppConfig.from_env()


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
    """Inizializza l'embedding con il provider configurato."""
    try:
        logger.info(f"Inizializzazione embeddings: provider={config.embedding_provider}, model={config.embedding_model}")
        embeddings = LLMFactory.create_embeddings(
            provider=config.embedding_provider,
            model=config.embedding_model,
            config=config
        )
        logger.info(f"Embeddings {config.embedding_provider} inizializzati con successo.")
        return embeddings
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
    """Carica i documenti PDF e verifica che non siano vuoti, con test di loader alternativi e supporto multimodale."""
    try:
        # Prova il loader principale per testo
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        logger.info("Tentativo di caricamento documenti con PyPDFDirectoryLoader...")
        loader = PyPDFDirectoryLoader(config.pdf_directory, glob="**/*.pdf")
        documents = loader.load()

        logger.info(f"PyPDFDirectoryLoader ha caricato {len(documents)} documenti")

        # Aggiungi processamento multimodale
        if config.enable_multimodal:
            logger.info("Avvio processamento multimodale per estrazione immagini/diagrammi...")
            from extensions.multimodal import MultimodalDocumentProcessor

            multimodal_processor = MultimodalDocumentProcessor(config)
            multimodal_documents = []
            pdf_files = [f for f in os.listdir(config.pdf_directory) if f.lower().endswith('.pdf')]

            for pdf_file in pdf_files:
                pdf_path = os.path.join(config.pdf_directory, pdf_file)
                try:
                    mm_docs, stats = multimodal_processor.process_pdf_document(pdf_path)
                    multimodal_documents.extend(mm_docs)
                    logger.info(f"PDF {pdf_file}: {stats.images_processed} immagini processate, {len(mm_docs)} documenti estratti")
                except Exception as e:
                    logger.warning(f"Errore processamento multimodale {pdf_file}: {e}")

            documents.extend(multimodal_documents)
            logger.info(f"Totale documenti (testo + multimodale): {len(documents)}")

        # Se non abbiamo documenti o sono insufficienti, proviamo loader alternativi
        if not documents or len(documents) < 5:
            logger.warning("Pochi documenti caricati, tentativo con loader alternativi...")
            try:
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

            print(f"\n📁 FILE NELLA DIRECTORY {config.pdf_directory}:")
            for filename in os.listdir(config.pdf_directory):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(config.pdf_directory, filename)
                    file_size = os.path.getsize(file_path) / 1024 / 1024
                    print(f"   - {filename} ({file_size:.1f} MB)")

            logger.error("Nessun contenuto significativo trovato nei PDF. Impossibile procedere.")
            raise ValueError("Nessun contenuto significativo trovato nei PDF. Controlla i file e riprova.")

        collected = gc.collect()
        if collected > 0:
            logger.info(f"Garbage collection dopo caricamento documenti: {collected} oggetti rimossi")

        return documents

    except Exception as e:
        logger.error(f"Errore nel caricamento dei documenti: {e}")
        print(f"Errore nel caricamento: {e}")
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        """
        Utilizza Chain-of-Thought reasoning per recuperare documenti rilevanti.

        Args:
            query: La domanda dell'utente
            run_manager: Callback manager (ignorato)

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
            if hasattr(reasoning_response, 'content'):
                reasoning_response = reasoning_response.content

            # STEP 2: Estrai le query di ricerca dal reasoning
            search_queries = self._extract_search_queries(reasoning_response, query)
            logger.info(f"Generate {len(search_queries)} query di ricerca specifiche")

            # STEP 3: Esegui ricerche sequenziali
            all_documents = []
            collected_info = []

            for i, search_query in enumerate(search_queries, 1):
                logger.info(f"Esecuzione ricerca {i}/{len(search_queries)}: {search_query[:80]}...")

                # Ricerca documenti per questa query specifica
                docs = self.base_retriever.invoke(search_query)

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
            return self.base_retriever.invoke(query)

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


class HyDERetriever(BaseRetriever):
    """
    Retriever che utilizza Hypothetical Document Embeddings (HyDE).
    Genera un documento ipotetico come risposta alla query, poi usa
    quell'embedding per la ricerca semantica nel vector store.
    """
    base_retriever: object
    config: object
    llm: object = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, base_retriever, config, **kwargs):
        super().__init__(base_retriever=base_retriever, config=config, **kwargs)

    def _get_llm(self):
        """Lazy initialization dell'LLM per la generazione ipotetica."""
        if self.llm is None:
            self.llm = LLMFactory.create_llm(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                config=self.config,
                temperature=0.5
            )
        return self.llm

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        """
        Genera un documento ipotetico per la query e usa il suo testo come query di ricerca.

        Args:
            query: La domanda dell'utente
            run_manager: Callback manager (ignorato)

        Returns:
            List[Document]: Documenti recuperati tramite embedding del documento ipotetico
        """
        logger.info(f"HyDE: generazione documento ipotetico per query: {query[:80]}...")
        try:
            llm = self._get_llm()
            hyde_prompt = PromptTemplate(
                input_variables=["question"],
                template=HYDE_PROMPT
            )
            hypothetical_response = llm.invoke(hyde_prompt.format(question=query))

            # Estrai il testo dalla risposta
            if hasattr(hypothetical_response, 'content'):
                search_text = hypothetical_response.content
            else:
                search_text = str(hypothetical_response)

            logger.info(f"HyDE: ricerca con documento ipotetico ({len(search_text)} char)")
            return self.base_retriever.invoke(search_text)

        except Exception as e:
            logger.error(f"Errore HyDE: {e}. Fallback a ricerca standard.")
            return self.base_retriever.invoke(query)


def get_retriever(config, embedding_provider, embedding_model):
    """Carica il retriever dalla cache. Fallisce se la cache non è valida."""
    logger.info("Tentativo di caricamento del retriever dalla cache...")

    # Aggiorna la configurazione per la sessione di embedding
    config.embedding_provider = embedding_provider
    config.embedding_model = embedding_model

    faiss_core_index_file = config.faiss_core_index_file
    docstore_path = config.docstore_path

    if not (os.path.exists(faiss_core_index_file) and os.path.getsize(faiss_core_index_file) > 0 and
            os.path.exists(docstore_path) and os.path.getsize(docstore_path) > 0):
        logger.error("Cache non trovata o non valida. Eseguire lo script 'build_hybrid_store.py' per creare l'indice.")
        return None

    try:
        base_embeddings = initialize_embeddings()
        vectorstore = FAISS.load_local(config.faiss_index_path, base_embeddings, allow_dangerous_deserialization=True)
        with open(docstore_path, "rb") as f:
            store = pickle.load(f)

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=config.child_chunk_size),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=config.parent_chunk_size),
        )
        logger.info("Retriever caricato con successo dalla cache.")
        return retriever
    except Exception as e:
        logger.error(f"Errore nel caricamento del retriever dalla cache: {e}")
        logger.error("La cache potrebbe essere corrotta o incompatibile. Eseguire 'build_hybrid_store.py' per rigenerarla.")
        return None

def get_qa_chain(retriever, retriever_type, llm_provider, llm_model, config):
    """Crea la catena di QA usando il retriever fornito e il modello LLM specificato."""
    logger.info(f"Creazione della catena QA con LLM: {llm_provider}/{llm_model} e retriever: {retriever_type}")

    # Aggiorna la configurazione per la sessione LLM
    config.llm_provider = llm_provider
    config.llm_model = llm_model

    # Applica i wrapper del retriever in base alla selezione
    final_retriever = retriever
    if retriever_type == "multi-query":
        logger.info("Applicazione del wrapper Multi-Query Retriever.")
        llm_for_mq = LLMFactory.create_llm(provider=llm_provider, model=llm_model, config=config, temperature=0)
        final_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm_for_mq)
    elif retriever_type == "chain-of-thought":
        logger.info("Modalità Chain-of-Thought Reasoning selezionata.")
        final_retriever = ChainOfThoughtRetriever(retriever, config)
    elif retriever_type == "hyde":
        logger.info("Modalità HyDE (Hypothetical Document Embeddings) selezionata.")
        final_retriever = HyDERetriever(retriever, config)

    try:
        llm = LLMFactory.create_llm(provider=llm_provider, model=llm_model, config=config, temperature=0.7)
        PROMPT = PromptTemplate(template=EXPERT_PROMPT, input_variables=["context", "question"])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | PROMPT
            | llm
            | StrOutputParser()
        )

        qa_chain = RunnablePassthrough.assign(
            context=(lambda x: final_retriever.invoke(x["query"])),
            question=(lambda x: x["query"]),
        ).assign(result=rag_chain_from_docs)

        logger.info("Catena QA LCEL creata con successo.")
        return qa_chain
    except Exception as e:
        logger.error(f"Errore nella creazione della catena QA: {e}")
        raise

