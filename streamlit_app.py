# /progetto_chatbot_pdf/streamlit_app.py

# Patch per risolvere il problema "There is no current event loop in thread" con Streamlit e Google
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import time
import os
from app import create_chatbot, PDF_DIRECTORY_PATH, VECTOR_STORE_PATH

# --- Configurazione della Pagina Streamlit ---
st.set_page_config(
    page_title="AI-syd-Gold Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI-syd-Gold: Assistente Tecnico per Cuscinetti")
st.caption("Fai domande sui documenti tecnici caricati. L'assistente risponderà basandosi sulla sua Knowledge Base.")

# --- Sidebar per la Configurazione ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Selezione del tipo di retriever
    retriever_type = st.radio(
        "Scegli la strategia di recupero:",
        ("standard", "hyde", "multi-query", "ensemble" ),
        index=0,
        help="""
        - **standard**: Usa ParentDocumentRetriever. Buon equilibrio tra contesto e precisione.
        - **hyde**: Genera una risposta ipotetica e la usa per cercare documenti simili.
        - **multi-query**: Genera varianti della tua domanda per una ricerca più ampia.
        - **ensemble**: Combina ricerca per parole chiave (BM25) e ricerca semantica (FAISS). Lento, non usa cache.
        """
    )
    
    st.info(f"Modalità selezionata: **{retriever_type}**")

    # Logica per gestire il cambio di retriever e informare l'utente
    if "active_retriever_type" not in st.session_state:
        st.session_state.active_retriever_type = retriever_type

    if st.session_state.active_retriever_type != retriever_type:
        st.session_state.active_retriever_type = retriever_type
        # Resetta la chat e informa l'utente del cambio
        st.session_state.messages = [
            {"role": "assistant", "content": "Ciao! Sono AI-syd-Gold. Come posso aiutarti oggi con i documenti tecnici?"},
            {"role": "assistant", "content": f"Modalità di ricerca aggiornata a **{retriever_type}**. La conversazione è stata resettata per coerenza."}
        ]
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Pulisci cronologia chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Ciao! Sono AI-syd-Gold. Come posso aiutarti oggi con i documenti tecnici?"}]
        st.rerun()

    st.markdown("---")
    st.header("📁 Knowledge Base")
    
    # Mostra lo stato delle directory
    pdf_dir = PDF_DIRECTORY_PATH
    cache_dir = VECTOR_STORE_PATH
    
    
    try:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        if pdf_files:
            st.success(f"Trovati {len(pdf_files)} PDF.")
            with st.expander("Mostra file PDF"):
                for pdf_file in pdf_files:
                    st.code(pdf_file, language=None)
        else:
            st.warning("Nessun file PDF trovato nella directory. L'app non funzionerà correttamente.")
    except FileNotFoundError:
        st.error(f"La directory PDF '{pdf_dir}' non esiste! Assicurati che sia creata e contenga i documenti.")

# --- Funzione per caricare il Chatbot (con cache) ---
@st.cache_resource(show_spinner="Inizializzazione del chatbot in corso... L'operazione potrebbe richiedere qualche minuto.")
def load_chatbot(retriever):
    """
    Carica e cachea la catena QA per evitare di ricaricarla ad ogni interazione.
    La cache viene invalidata se il tipo di retriever cambia.
    """
    try:
        return create_chatbot(retriever_type=retriever)
    except Exception as e:
        st.error(f"Errore critico durante l'inizializzazione del chatbot: {e}")
        st.error("Controlla i log del terminale per dettagli. Verifica la chiave API e i file PDF.")
        return None

# Carica il chatbot
qa_chain = load_chatbot(retriever=retriever_type)

if qa_chain is None:
    st.warning("Il chatbot non è stato inizializzato. Impossibile procedere.")
    st.stop()

# --- Gestione della Cronologia della Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Sono AI-syd-Gold. Come posso aiutarti oggi con i documenti tecnici?"}]

# Mostra i messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input dell'Utente e Generazione della Risposta ---
if prompt := st.chat_input("Fai la tua domanda qui..."):
    # Aggiungi il messaggio dell'utente alla cronologia
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genera e mostra la risposta dell'assistente
    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la risposta nei documenti..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                
                # 1. Mostra la risposta principale
                result = response.get("result", "Non sono riuscito a trovare una risposta.")
                st.markdown(result)

                # 2. Prepara il testo delle fonti per la cronologia (versione semplice)
                sources_text_for_history = ""
                source_documents = response.get("source_documents")
                if source_documents:
                    # Crea un elenco testuale semplice per la cronologia
                    sources_text_for_history = "\n\n---\n*Fonti utilizzate:*"
                    source_files = sorted(list(set(os.path.basename(doc.metadata.get('source', 'N/A')) for doc in source_documents)))
                    sources_text_for_history += "\n- " + "\n- ".join(source_files) if source_files else " Nessuna fonte specifica identificata."

                    # 3. Mostra le fonti in modo interattivo e dettagliato nell'interfaccia
                    st.markdown("---")
                    with st.expander("Vedi fonti e contesto utilizzato"):
                        # Raggruppa i documenti per file sorgente per una visualizzazione pulita
                        sources_by_file = {}
                        for doc in source_documents:
                            source_name = os.path.basename(doc.metadata.get('source', 'Sconosciuto'))
                            if source_name not in sources_by_file:
                                sources_by_file[source_name] = []
                            sources_by_file[source_name].append(doc)
                        
                        for filename, docs in sorted(sources_by_file.items()):
                            st.markdown(f"#### 📄 {filename}")
                            for i, doc in enumerate(docs):
                                st.info(f"**Contesto recuperato {i+1}:**\n\n" + doc.page_content)
                
                full_response_for_history = result + sources_text_for_history

            except Exception as e:
                st.error(f"Si è verificato un errore durante l'elaborazione della tua domanda: {e}")
                full_response_for_history = "Mi dispiace, si è verificato un errore. Riprova."
                st.markdown(full_response_for_history)

    # Aggiungi la risposta completa dell'assistente alla cronologia
    st.session_state.messages.append({"role": "assistant", "content": full_response_for_history})