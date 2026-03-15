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
from config.system_prompt import COMMUNICATION_STYLES

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
CHAT_HISTORY_TURNS = 10  # Numero di turni (coppie user/assistant) da passare al LLM come contesto

def init_session_state():
    defaults = {
        'pdf_path': None,
        'pdf_page': 0,
        'messages': [{"role": "assistant", "content": "Ciao! Sono BearX. Come posso aiutarti?"}],
        'communication_style': 'Consultant',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_context_prefix(messages: list, max_turns: int = CHAT_HISTORY_TURNS) -> str:
    """
    Costruisce un prefisso testuale con gli ultimi max_turns turni della conversazione.
    Viene anteposto alla query corrente per dare al LLM memoria della sessione.
    Esclude il messaggio iniziale di benvenuto e limita a max_turns coppie user/assistant.
    """
    # Filtra solo messaggi user/assistant (esclude il benvenuto iniziale statico)
    conversation = [m for m in messages if not (m["role"] == "assistant" and "Sono BearX" in m["content"] and len(m["content"]) < 80)]
    if not conversation:
        return ""
    # Prendi gli ultimi max_turns * 2 messaggi (ogni turno = 1 user + 1 assistant)
    recent = conversation[-(max_turns * 2):]
    lines = ["[CONVERSAZIONE PRECEDENTE — usa questi dati se rilevanti per la domanda attuale]"]
    for msg in recent:
        role_label = "Utente" if msg["role"] == "user" else "BearX"
        lines.append(f"{role_label}: {msg['content'][:500]}")  # Tronca messaggi molto lunghi
    lines.append("[FINE CONVERSAZIONE PRECEDENTE]")
    return "\n".join(lines)

init_session_state()

# --- Definizione della Finestra Modale ---
path = st.session_state.get('pdf_path')
pdf_title = os.path.basename(path) if path else ""
modal = Modal(title=pdf_title, key="pdf_viewer_modal", max_width=1200)

st.markdown("""
<style>
    /* Layout */
    .block-container { padding-top: 1rem !important; }
    section[data-testid="stSidebar"] > div { padding-top: 0.3rem !important; }

    /* Messaggi chat — paragrafi */
    div[data-testid="stChatMessage"] .stMarkdown p {
        line-height: 1.75;
        margin-bottom: 0.6rem;
    }

    /* Titoli nelle risposte */
    div[data-testid="stChatMessage"] .stMarkdown h1,
    div[data-testid="stChatMessage"] .stMarkdown h2,
    div[data-testid="stChatMessage"] .stMarkdown h3 {
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    /* Liste */
    div[data-testid="stChatMessage"] .stMarkdown ul,
    div[data-testid="stChatMessage"] .stMarkdown ol {
        padding-left: 1.4rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stChatMessage"] .stMarkdown li {
        margin-bottom: 0.3rem;
        line-height: 1.65;
    }

    /* Tabelle */
    div[data-testid="stChatMessage"] .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    div[data-testid="stChatMessage"] .stMarkdown th {
        background-color: #2a4a6b;
        color: #ffffff;
        padding: 8px 12px;
        text-align: left;
        border: 1px solid #3d6090;
    }
    div[data-testid="stChatMessage"] .stMarkdown td {
        padding: 7px 12px;
        border: 1px solid #3d3d3d;
    }
    div[data-testid="stChatMessage"] .stMarkdown tr:nth-child(even) td {
        background-color: rgba(255,255,255,0.04);
    }

    /* Codice inline */
    div[data-testid="stChatMessage"] .stMarkdown code {
        background-color: #1e2a3a;
        color: #7ec8e3;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.88rem;
    }

    /* Blocchi di codice */
    div[data-testid="stChatMessage"] .stMarkdown pre {
        background-color: #141e2b;
        border-left: 3px solid #4a8fcc;
        padding: 12px 16px;
        border-radius: 6px;
        overflow-x: auto;
    }

    /* Grassetto */
    div[data-testid="stChatMessage"] .stMarkdown strong {
        color: #90caf9;
    }

    /* Separatori */
    div[data-testid="stChatMessage"] .stMarkdown hr {
        border-color: #3d3d3d;
        margin: 1rem 0;
    }

    /* Blockquote */
    div[data-testid="stChatMessage"] .stMarkdown blockquote {
        border-left: 3px solid #4a8fcc;
        padding: 4px 12px;
        margin: 0.6rem 0;
        color: #aab8c8;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title("☸️ BearX: a bearings expert")
st.caption("Fai domande sul mondo dei cuscinetti, proverò a rispondere basandomi sulla mia Knowledge Base.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurazione")

    # --- Communication Style ---
    style_options = list(COMMUNICATION_STYLES.keys())
    style_descriptions = {
        "Expert": "Diretto e tecnico, nessuna spiegazione extra",
        "Consultant": "Bilanciato, spiega il ragionamento (default)",
        "Teacher": "Didattico, spiega i principi e i termini",
    }
    selected_style = st.selectbox(
        "💬 Stile di Risposta:",
        style_options,
        index=style_options.index(st.session_state.get("communication_style", "Consultant")),
    )
    st.info(f"**{selected_style}** — {style_descriptions[selected_style]}")
    st.session_state.communication_style = selected_style

    # --- LLM Models ---
    _label_to_model = LLMFactory.GOOGLE_MODEL_LABELS
    _model_to_label = {v: k for k, v in _label_to_model.items()}
    _default_label = _model_to_label.get(config.llm_model, "Lite")
    _label_options = list(_label_to_model.keys())
    selected_label = st.selectbox(
        "🤖 Modello Reasoning:",
        _label_options,
        index=_label_options.index(_default_label),
        help="Lite: veloce e leggero | Flash: bilanciato (default) | Pro: massima qualità",
    )
    selected_model = _label_to_model[selected_label]
    selected_provider = "google"

    selected_embedding_provider = config.embedding_provider
    selected_embedding_model = config.embedding_model

    st.info(f"**Gemini {selected_label}** — `{selected_model}`")
    st.markdown("<hr style='margin: 0.4rem 0; border-color: #3d3d3d;'>", unsafe_allow_html=True)

    # --- Advanced Settings (collapsible) ---
    with st.expander("🔧 Impostazioni Avanzate"):
        st.caption("Strategia di retrieval — Auto usa la migliore in base alla query.")
        retriever_type = st.radio(
            "Strategia di Recupero:",
            ("standard", "hyde", "multi-query"),
            index=0,
            help="standard: ricerca diretta | hyde: genera risposta ipotetica per la ricerca | multi-query: riformula la query in varianti multiple",
        )
        st.info(f"Strategia: **{retriever_type}**")

    if st.session_state.get("active_retriever_type") != retriever_type or st.session_state.get("active_llm_model") != selected_model:
        st.session_state.active_retriever_type = retriever_type
        st.session_state.active_llm_model = selected_model
        st.session_state.messages = [{"role": "assistant", "content": "Ciao! Sono BearX.  Come posso aiutarti?"}]

    if st.button("🗑️ Pulisci cronologia chat"):
        init_session_state()
        st.rerun()

    st.markdown("<hr style='margin: 0.4rem 0; border-color: #3d3d3d;'>", unsafe_allow_html=True)
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
def load_qa_chain(_retriever, retriever_type, llm_provider, llm_model, communication_style):
    t0 = time.time()
    result = get_qa_chain(_retriever, retriever_type, llm_provider, llm_model, config, communication_style)
    elapsed = time.time() - t0
    st.sidebar.caption(f"⏱ LLM pronto in {elapsed:.1f}s")
    return result

base_retriever = load_retriever(selected_embedding_provider, selected_embedding_model)
qa_chain = None
if base_retriever:
    try:
        qa_chain = load_qa_chain(base_retriever, retriever_type, selected_provider, selected_model, selected_style)
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
            "Olio i circuiti...",
            "Applico il precarico corretto ai neuroni...",
            "Verifico il gioco radiale della risposta...",
            "Controllo che la gabbia non si deformi sotto pressione...",
            "Calcolo la durata L10 della mia pazienza...",
            "Cerco freneticamente nel catalogo SKF pagina 347...",
            "Applico grasso NLGI 2 ai cuscinetti del processore...",
            "Verifico che la velocità di rotazione non superi il limite termico...",
            "Misuro la rugosità Ra della domanda con il profilometro...",
            "Stringo i bulloni della risposta al momento corretto...",
            "Controllo che non ci siano cricche da fatica nei dati...",
            "Eseguo l'analisi agli elementi finiti della query...",
            "Seleziono la tolleranza ISO adeguata per questa risposta...",
            "Verifico che il fattore di sicurezza sia almeno 1.5...",
            "Consulto la ISO 281 come se fosse la Bibbia...",
            "Riduco il rumore NVH della risposta...",
            "Bilancio dinamicamente i concetti prima di lanciarli...",
            "Misuro la durezza Brinell dell'argomentazione...",
            "Scaldo il forno per il trattamento termico dei dati...",
            "Verifico che il fit albero-mozzo sia davvero H7/k6...",
            "Eseguo il rodaggio della risposta a bassa velocità...",
            "Monitoro la temperatura con la termocamera — nulla di grave...",
            "Analizzo lo spettro FFT dei dati recuperati...",
            "Controllo che il coefficiente di attrito sia accettabile...",
            "Cerco l'O-ring giusto tra 847 varianti standard...",
            "Porto il viscosimetro a temperatura di esercizio...",
        ]
        spinner_text = random.choice(spinner_messages)
        with st.spinner(spinner_text):
            try:
                # Costruisci il contesto della sessione con i turni precedenti
                context_prefix = build_context_prefix(st.session_state.messages)
                query_with_context = f"{context_prefix}\n\n{prompt}" if context_prefix else prompt

                response = qa_chain.invoke({"query": query_with_context})
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
                import logging
                logging.error(f"BearX chat error: {traceback.format_exc()}")
                error_type = type(e).__name__
                if "timeout" in str(e).lower() or "deadline" in str(e).lower():
                    user_msg = "La risposta ha impiegato troppo tempo. Prova a riformulare la domanda in modo più conciso."
                elif "quota" in str(e).lower() or "rate" in str(e).lower():
                    user_msg = "Limite di utilizzo API raggiunto. Attendi qualche secondo e riprova."
                elif "context" in str(e).lower() or "token" in str(e).lower():
                    user_msg = "La sessione è diventata troppo lunga. Usa il tasto 'Pulisci cronologia chat' per ricominciare."
                else:
                    user_msg = f"Si è verificato un errore imprevisto ({error_type}). Riprova o pulisci la cronologia."
                st.error(user_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {user_msg}"})
