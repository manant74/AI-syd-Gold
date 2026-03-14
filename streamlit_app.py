# /progetto_chatbot_pdf/streamlit_app.py

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import time
import random
import fitz  # PyMuPDF
from PIL import Image
import io
from streamlit_modal import Modal

from app import get_retriever, get_qa_chain
from config import AppConfig
from config.llm_providers import LLMFactory

config = AppConfig.from_env()

# --- Funzioni per PDF Viewer ---
@st.cache_data
def render_pdf_page(_pdf_path, page_num, zoom):
    try:
        doc = fitz.open(_pdf_path)
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        return img_data
    except Exception as e:
        st.error(f"Errore rendering PDF: {e}")
        return None

# --- Configurazione Pagina ---
st.set_page_config(page_title="BearX Chatbot", page_icon="☸️", layout="wide", initial_sidebar_state="expanded")

# --- Inizializzazione Session State ---
def init_session_state():
    defaults = {
        'pdf_path': None,
        'pdf_page': 0,
        'messages': [{"role": "assistant", "content": "Ciao! Sono BearX. Come posso aiutarti?"}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Definizione della Finestra Modale ---
path = st.session_state.get('pdf_path')
pdf_title = os.path.basename(path) if path else ""
modal = Modal(title=pdf_title, key="pdf_viewer_modal", max_width=1200)

st.title("☸️ BearX: a bearings expert")
st.caption("Fai domande sul mondo dei cuscinetti, proverò a rispondere basandomi sulla mia Knowledge Base.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    st.subheader("🤖 Modelli LLM")
    
    available_models = LLMFactory.get_available_models()
    llm_providers = [p for p, m in available_models.items() if m["llm"]]
    try: provider_index = llm_providers.index(config.llm_provider)
    except ValueError: provider_index = 0
    selected_provider = st.selectbox("Provider:", llm_providers, index=provider_index)

    available_llm_models = available_models[selected_provider]["llm"]
    try: model_index = available_llm_models.index(config.llm_model)
    except ValueError: model_index = 0
    selected_model = st.selectbox("Modello:", available_llm_models, index=model_index)

    selected_embedding_provider = config.embedding_provider
    selected_embedding_model = config.embedding_model

    st.info(f"**LLM:** {selected_provider}/{selected_model}")
    st.markdown("---")

    st.subheader("🔍 Strategia di Recupero")
    retriever_type = st.radio("Modalità:", ("standard", "hyde", "multi-query", "chain-of-thought"), index=0)
    st.info(f"Modalità selezionata: **{retriever_type}**")

    if st.session_state.get("active_retriever_type") != retriever_type or st.session_state.get("active_llm_model") != selected_model:
        st.session_state.active_retriever_type = retriever_type
        st.session_state.active_llm_model = selected_model
        st.session_state.messages = [{"role": "assistant", "content": "Configurazione aggiornata. La chat è stata resettata. Come posso aiutarti?"}]

    st.markdown("---")
    if st.button("🗑️ Pulisci cronologia chat"):
        init_session_state()
        st.rerun()

    st.markdown("---")
    st.header("📁 Knowledge Base")
    pdf_dir = config.pdf_directory
    try:
        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')])
        if pdf_files:
            st.success(f"Trovati {len(pdf_files)} PDF.")
            with st.expander("Mostra e seleziona file PDF"):
                st.markdown("<style>div[data-testid=\"stExpander\"] div[data-testid=\"stVerticalBlock\"] button {text-align: left !important; justify-content: flex-start !important;}</style>", unsafe_allow_html=True)
                for pdf_file in pdf_files:
                    if st.button(pdf_file, key=f"pdf_{pdf_file}", use_container_width=True):
                        if st.session_state.pdf_path != os.path.join(pdf_dir, pdf_file):
                            st.session_state.pdf_path = os.path.join(pdf_dir, pdf_file)
                            st.session_state.pdf_page = 0
                            render_pdf_page.clear()
                        modal.open()
    except FileNotFoundError:
        st.error(f"La directory PDF '{pdf_dir}' non esiste!")

# --- Funzioni Caricamento Chatbot ---
@st.cache_resource(show_spinner="Caricamento indice vettoriale...")
def load_retriever(embedding_provider, embedding_model):
    t0 = time.time()
    result = get_retriever(config, embedding_provider, embedding_model)
    elapsed = time.time() - t0
    if result:
        st.sidebar.caption(f"⏱ Indice caricato in {elapsed:.1f}s")
    return result

@st.cache_resource(show_spinner="Inizializzazione del modello LLM...")
def load_qa_chain(_retriever, retriever_type, llm_provider, llm_model):
    t0 = time.time()
    result = get_qa_chain(_retriever, retriever_type, llm_provider, llm_model, config)
    elapsed = time.time() - t0
    st.sidebar.caption(f"⏱ LLM pronto in {elapsed:.1f}s")
    return result

base_retriever = load_retriever(selected_embedding_provider, selected_embedding_model)
qa_chain = None
if base_retriever:
    try:
        qa_chain = load_qa_chain(base_retriever, retriever_type, selected_provider, selected_model)
    except ValueError as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api_key" in error_msg.lower():
            st.sidebar.error(f"Chiave API mancante: {error_msg}\n\nAggiungi la variabile al tuo file `.env`")
        else:
            st.sidebar.error(f"Errore configurazione: {error_msg}")
    except Exception as e:
        st.sidebar.error(f"Errore inizializzazione LLM ({selected_provider}): {e}")

# --- Contenuto della Finestra Modale ---
if modal.is_open() and st.session_state.pdf_path:
    with modal.container():
        st.markdown(f"### {os.path.basename(st.session_state.pdf_path)}")
        try:
            pdf_path = st.session_state.pdf_path
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            nav_cols = st.columns([2, 2, 2, 2, 2])
            with nav_cols[0]:
                st.markdown(f" ")
            with nav_cols[1]:
                st.markdown(f"<div style='margin-top: 8px;'>Pagina {st.session_state.pdf_page + 1}/{total_pages}</div>", unsafe_allow_html=True)
            with nav_cols[2]:
                page_num_input = st.number_input("Pagina", min_value=1, max_value=total_pages, value=st.session_state.pdf_page + 1, key=f"page_input_{pdf_path}", label_visibility="collapsed")
                if page_num_input != st.session_state.pdf_page + 1:
                    st.session_state.pdf_page = page_num_input - 1
                    st.rerun()
            with nav_cols[3]:
                with open(pdf_path, "rb") as f:
                    st.download_button("Apri / Scarica PDF", f.read(), os.path.basename(pdf_path), "application/pdf", use_container_width=True)
            with nav_cols[4]:
                st.markdown(f" ")
                
            img_data = render_pdf_page(pdf_path, st.session_state.pdf_page, 1.5)
            if img_data:
                _, img_col, _ = st.columns([1, 6, 1])
                with img_col:
                    st.image(Image.open(io.BytesIO(img_data)))
            else:
                st.error("Impossibile visualizzare la pagina.")
        except Exception as e:
            st.error(f"Impossibile caricare il PDF: {e}")

# --- Layout Principale della Chat ---
if qa_chain is None:
    st.error("Il chatbot non è stato inizializzato. Esegui lo script `build_hybrid_store.py`.")
    st.stop()

# Visualizza la cronologia della chat
for message in st.session_state.messages:
    avatar = "🐻" if message["role"] == "assistant" else "🧑‍🔧"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        # Se il messaggio è dall'assistente e ha fonti, mostrale
        if message["role"] == "assistant" and "source_documents" in message and message["source_documents"]:
            with st.expander("Vedi fonti e contesto utilizzato"):
                for doc in message["source_documents"]:
                    doc_source = doc.metadata.get('source', 'Sconosciuto')
                    doc_page = doc.metadata.get('page', -1)
                    if st.button(f"📄 {os.path.basename(doc_source)}, Pagina {doc_page + 1}", key=f"src_{doc_source}_{doc_page}_{message['content'][:10]}"):
                        st.session_state.pdf_path = doc_source
                        st.session_state.pdf_page = doc_page
                        render_pdf_page.clear()
                        modal.open()
                    st.info(f"**Contesto:**\n{doc.page_content}")

# Gestisci l'input dell'utente
if prompt := st.chat_input("Fai la tua domanda qui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🧑‍🔧"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🐻"):
        spinner_messages = [
            "Consulto gli antichi manuali...",
            "Lubrifico gli ingranaggi della conoscenza...",
            "Calcolo le tolleranze della tua domanda...",
            "Verifico il gioco assiale dei dati...",
            "Scaldo i motori del ragionamento...",
            "Allineo i cuscinetti semantici...",
            "Sfoglio i cataloghi tecnici...",
            "Preparo il calibro per la risposta...",
            "Analizzo la rugosità della query...",
            "Metto in rotazione i dati...",
            "Controllo la viscosità delle informazioni...",
            "Assemblo i componenti della risposta...",
            "Evito il grippaggio del server...",
            "Cerco la giusta coppia di serraggio...",
            "Lucido le sfere della sapienza...",
            "Avvio la turbina della ricerca...",
            "Calibro la risposta...",
            "Verifico la durezza Rockwell dei fatti...",
            "Olio i circuiti..."
        ]
        spinner_text = random.choice(spinner_messages)
        with st.spinner(spinner_text):
            try:
                response = qa_chain.invoke({"query": prompt})
                result = response.get("result", "Non sono riuscito a trovare una risposta.")
                source_documents = response.get("context")
                
                # Aggiungi la risposta e le fonti alla cronologia
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result, 
                    "source_documents": source_documents
                })
                st.rerun() # riesegui per mostrare il nuovo messaggio e le fonti

            except Exception as e:
                import traceback
                err_details = traceback.format_exc()
                st.error(f"Si è verificato un errore: {e}")
                st.error(err_details)
                st.session_state.messages.append({"role": "assistant", "content": "Mi dispiace, si è verificato un errore."})
