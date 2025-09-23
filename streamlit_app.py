# /progetto_chatbot_pdf/streamlit_app.py

# Patch per risolvere il problema "There is no current event loop in thread" con Streamlit e Google
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import time
import os
import fitz  # PyMuPDF per rendering PDF
from PIL import Image
import io
from app import create_chatbot
from config import AppConfig
from llm_providers import LLMFactory

# Configurazione centralizzata
config = AppConfig.from_env()

# --- Funzioni per PDF Viewer ---
@st.cache_data
def render_pdf_page(pdf_path, page_num=0, zoom=1.5):
    """Renderizza una pagina PDF come immagine (con cache)."""
    try:
        doc = fitz.open(pdf_path)
        if page_num >= doc.page_count:
            page_num = 0

        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()

        return img_data  # Ritorna bytes per il caching
    except Exception as e:
        st.error(f"Errore nel caricamento PDF: {e}")
        return None

def display_pdf_page(pdf_path, page_num=0, zoom=1.5):
    """Wrapper per visualizzare la pagina PDF."""
    img_data = render_pdf_page(pdf_path, page_num, zoom)
    if img_data:
        return Image.open(io.BytesIO(img_data))
    return None

def show_pdf_panel(container):
    """Mostra il pannello PDF in un container."""
    if not st.session_state.get('show_pdf_viewer', False):
        return

    pdf_filename = st.session_state.get('pdf_to_show', '')
    if not pdf_filename:
        return

    with container:
        # Header del pannello PDF
        col_header, col_close = st.columns([4, 1])

        with col_header:
            st.markdown(f"### 📄 {pdf_filename}")

        with col_close:
            if st.button("❌", help="Chiudi PDF"):
                st.session_state.show_pdf_viewer = False
                st.rerun()

        # Trova il path completo del PDF
        pdf_path = os.path.join(config.pdf_directory, pdf_filename)

        if not os.path.exists(pdf_path):
            st.error(f"File PDF non trovato: {pdf_filename}")
            return

        try:
            # Informazioni PDF
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()

            # Controlli in colonne
            col_page, col_zoom = st.columns(2)

            with col_page:
                # Slider per selezione pagina
                current_page = st.slider(
                    f"Pagina (1-{total_pages}):",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.get('pdf_current_page', 1),
                    key='pdf_page_slider'
                )

            with col_zoom:
                # Controllo zoom
                zoom_level = st.slider(
                    "Zoom:",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.2,
                    step=0.1,
                    key='pdf_zoom_slider'
                )

            # Aggiorna pagina corrente
            st.session_state.pdf_current_page = current_page

            # Renderizza pagina
            with st.spinner("Caricamento pagina..."):
                img = display_pdf_page(pdf_path, current_page - 1, zoom_level)
                if img:
                    st.image(img, caption=f"Pagina {current_page}", use_column_width=True)
                else:
                    st.error("Impossibile caricare la pagina")

        except Exception as e:
            st.error(f"Errore: {e}")

# --- Configurazione della Pagina Streamlit ---
st.set_page_config(
    page_title="BearX Chatbot",
    page_icon="☸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializza session state per PDF viewer
if 'show_pdf_viewer' not in st.session_state:
    st.session_state.show_pdf_viewer = False
if 'pdf_to_show' not in st.session_state:
    st.session_state.pdf_to_show = None
if 'pdf_current_page' not in st.session_state:
    st.session_state.pdf_current_page = 1

st.title("☸️ BearX: a bearings expert")
st.caption("Fai domande sul mondo dei cuscinetti, proverò a rispondere basandomi sulla mia Knowledge Base.")

# --- Sidebar per la Configurazione ---
with st.sidebar:
    st.header("⚙️ Configurazione")

    # === SEZIONE MODELLI LLM ===
    st.subheader("🤖 Modelli LLM")

    # Ottieni i modelli disponibili
    available_models = LLMFactory.get_available_models()

    # Filtra solo i provider che hanno modelli LLM
    llm_providers = [provider for provider, models in available_models.items() if models["llm"]]

    # Selectbox per il provider
    selected_provider = st.selectbox(
        "Provider:",
        llm_providers,
        index=llm_providers.index(config.llm_provider) if config.llm_provider in llm_providers else 0,
        help="Scegli il provider di modelli AI"
    )

    # Selectbox per il modello (dinamico basato sul provider)
    available_llm_models = available_models[selected_provider]["llm"]
    current_model = config.llm_model if config.llm_model in available_llm_models else available_llm_models[0]

    selected_model = st.selectbox(
        "Modello:",
        available_llm_models,
        index=available_llm_models.index(current_model),
        help=f"Modelli disponibili per {selected_provider}"
    )

    # Embeddings fissi su Google per ora (da sostituire con HuggingFace quando quota è resettata)
    selected_embedding_provider = "google"
    selected_embedding_model = "models/embedding-001"

    # Mostra configurazione attuale
    st.info(f"**LLM:** {selected_provider}/{selected_model}")

    st.markdown("---")

    # === SEZIONE RETRIEVER ===
    st.subheader("🔍 Strategia di Recupero")

    # Selezione del tipo di retriever
    retriever_type = st.radio(
        "Modalità:",
        ("standard", "hyde", "multi-query", "ensemble", "chain-of-thought"),
        index=0,
        help="""
        - **standard**: Usa ParentDocumentRetriever. Buon equilibrio tra contesto e precisione.
        - **hyde**: Genera una risposta ipotetica e la usa per cercare documenti simili.
        - **multi-query**: Genera varianti della tua domanda per una ricerca più ampia.
        - **ensemble**: Combina ricerca per parole chiave (BM25) e ricerca semantica (FAISS). Lento, non usa cache.
        - **chain-of-thought**: Analizza la domanda step-by-step e pianifica una strategia di ricerca ottimale. Ideale per problemi complessi.
        """
    )

    st.info(f"Modalità selezionata: **{retriever_type}**")

    # Logica per gestire i cambi di configurazione
    # Inizializza lo stato della sessione se non esiste
    if "active_retriever_type" not in st.session_state:
        st.session_state.active_retriever_type = retriever_type
    if "active_llm_provider" not in st.session_state:
        st.session_state.active_llm_provider = selected_provider
    if "active_llm_model" not in st.session_state:
        st.session_state.active_llm_model = selected_model
    if "active_embedding_provider" not in st.session_state:
        st.session_state.active_embedding_provider = selected_embedding_provider
    if "active_embedding_model" not in st.session_state:
        st.session_state.active_embedding_model = selected_embedding_model

    # Verifica se qualche configurazione è cambiata
    config_changed = False
    change_messages = []

    if st.session_state.active_retriever_type != retriever_type:
        st.session_state.active_retriever_type = retriever_type
        change_messages.append(f"Modalità di ricerca aggiornata a **{retriever_type}**")
        config_changed = True

    if st.session_state.active_llm_provider != selected_provider:
        st.session_state.active_llm_provider = selected_provider
        change_messages.append(f"Provider LLM cambiato a **{selected_provider}**")
        config_changed = True

    if st.session_state.active_llm_model != selected_model:
        st.session_state.active_llm_model = selected_model
        change_messages.append(f"Modello LLM cambiato a **{selected_model}**")
        config_changed = True

    if st.session_state.active_embedding_provider != selected_embedding_provider:
        st.session_state.active_embedding_provider = selected_embedding_provider
        change_messages.append(f"Provider Embeddings cambiato a **{selected_embedding_provider}**")
        config_changed = True

    if st.session_state.active_embedding_model != selected_embedding_model:
        st.session_state.active_embedding_model = selected_embedding_model
        change_messages.append(f"Modello Embeddings cambiato a **{selected_embedding_model}**")
        config_changed = True

    # Se la configurazione è cambiata, resetta la chat e ricarica la pagina
    if config_changed:
        # Resetta la chat e informa l'utente dei cambiamenti
        st.session_state.messages = [
            {"role": "assistant", "content": "Ciao! Sono BearX. Come posso aiutarti oggi sui cuscinetti?"},
            {"role": "assistant", "content": f"Configurazione aggiornata:\n" + "\n".join([f"• {msg}" for msg in change_messages]) + "\n\nLa conversazione è stata resettata per coerenza."}
        ]

        # Ricarica la pagina per applicare i nuovi parametri
        # La cache automatica di Streamlit userà i nuovi parametri come chiave
        st.rerun()

    st.markdown("---")

    # Pulsante di controllo
    if st.button("🗑️ Pulisci cronologia chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Ciao! Sono BearX. Come posso aiutarti oggi con i cuscinetti?"}]
        st.rerun()

    st.markdown("---")
    st.header("📁 Knowledge Base")
    
    # Mostra lo stato delle directory
    pdf_dir = config.pdf_directory
    cache_dir = config.vector_store_directory
    
    
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
def load_chatbot(retriever, llm_provider, llm_model, embedding_provider, embedding_model):
    """
    Carica e cachea la catena QA per evitare di ricaricarla ad ogni interazione.
    La cache viene invalidata se il tipo di retriever o modelli cambiano.
    La chiave di cache include tutti i parametri per garantire il ricaricamento quando cambiano.
    """
    try:
        # Crea una configurazione temporanea per questo chatbot
        import copy
        temp_config = copy.deepcopy(config)
        temp_config.llm_provider = llm_provider
        temp_config.llm_model = llm_model

        # Aggiorna temporaneamente la configurazione globale
        original_llm_provider = config.llm_provider
        original_llm_model = config.llm_model
        original_embedding_provider = config.embedding_provider
        original_embedding_model = config.embedding_model

        config.llm_provider = llm_provider
        config.llm_model = llm_model
        config.embedding_provider = embedding_provider
        config.embedding_model = embedding_model

        chatbot = create_chatbot(retriever_type=retriever)

        # Ripristina la configurazione originale
        config.llm_provider = original_llm_provider
        config.llm_model = original_llm_model
        config.embedding_provider = original_embedding_provider
        config.embedding_model = original_embedding_model

        return chatbot
    except Exception as e:
        st.error(f"Errore critico durante l'inizializzazione del chatbot: {e}")
        st.error("Controlla i log del terminale per dettagli. Verifica la chiave API e i file PDF.")
        return None

# Carica il chatbot con i parametri selezionati DOPO aver gestito i cambiamenti
# La cache di Streamlit userà automaticamente i parametri come chiave
qa_chain = load_chatbot(
    retriever=retriever_type,
    llm_provider=selected_provider,
    llm_model=selected_model,
    embedding_provider=selected_embedding_provider,
    embedding_model=selected_embedding_model
)

if qa_chain is None:
    st.warning("Il chatbot non è stato inizializzato. Impossibile procedere.")
    st.stop()

# --- Gestione della Cronologia della Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Sono BearX. Come posso aiutarti oggi con i documenti tecnici?"}]

# Layout principale: chat a sinistra, PDF viewer a destra (quando attivo)
if st.session_state.get('show_pdf_viewer', False):
    col_chat, col_pdf = st.columns([2, 1])
else:
    col_chat = st.container()
    col_pdf = None

with col_chat:
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
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(f"#### 📄 {filename}")

                                with col2:
                                    # Bottone per aprire PDF nella sidebar
                                    if st.button(f"📖 Visualizza", key=f"view_pdf_{filename}"):
                                        st.session_state.pdf_to_show = filename
                                        st.session_state.show_pdf_sidebar = True
                                        # Trova la prima pagina con contenuto di questo file
                                        first_page = None
                                        for doc in docs:
                                            if doc.metadata.get('page'):
                                                try:
                                                    page_num = int(doc.metadata.get('page', 1))
                                                    if first_page is None or page_num < first_page:
                                                        first_page = page_num
                                                except:
                                                    pass
                                        if first_page:
                                            st.session_state.pdf_current_page = first_page
                                        st.rerun()

                                for i, doc in enumerate(docs):
                                    page_num = doc.metadata.get('page', 'N/A')
                                    page_info = f" (Pagina {page_num})" if page_num != 'N/A' else ""

                                    # Container per contenuto e bottone pagina specifica
                                    with st.container():
                                        col_content, col_page_btn = st.columns([4, 1])

                                        with col_content:
                                            st.info(f"**Contesto recuperato {i+1}:**{page_info}\n\n" + doc.page_content)

                                        with col_page_btn:
                                            # Bottone per aprire direttamente questa pagina
                                            if page_num != 'N/A' and st.button(f"Pag. {page_num}", key=f"goto_page_{filename}_{i}"):
                                                st.session_state.pdf_to_show = filename
                                                st.session_state.show_pdf_sidebar = True
                                                st.session_state.pdf_current_page = int(page_num)
                                                st.rerun()
                    
                    full_response_for_history = result + sources_text_for_history

                except Exception as e:
                    st.error(f"Si è verificato un errore durante l'elaborazione della tua domanda: {e}")
                    full_response_for_history = "Mi dispiace, si è verificato un errore. Riprova."
                    st.markdown(full_response_for_history)

    # Aggiungi la risposta completa dell'assistente alla cronologia
    st.session_state.messages.append({"role": "assistant", "content": full_response_for_history})

# --- Visualizzazione PDF Sidebar ---
# Questa funzione deve essere chiamata alla fine per gestire il PDF viewer
## show_pdf_sidebar()