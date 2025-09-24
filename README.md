# AI-syd-Gold - Advanced PDF Chatbot for Mechanical Engineering

🚀 **Un sistema RAG avanzato specializzato in ingegneria meccanica e cuscinetti**, che utilizza tecnologie multimodali, multiple strategie di retrieval, e supporto per diversi provider di AI.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com)

## 🌟 Caratteristiche Principali

### 🧠 **Multi-Provider AI Support**
- **Google**: Gemini 2.0, 2.5 Flash, 2.5 Pro
- **OpenAI**: GPT-4o, o1-preview, o1-mini
- **Anthropic**: Claude 3.5 Haiku, Claude 4 Sonnet/Opus
- **Ollama**: Modelli locali (offline)

### 🖼️ **Elaborazione Multimodale**
- **OCR Integrato**: Estrazione testo da immagini e diagrammi tecnici
- **Riconoscimento Intelligente**: Classificazione automatica di tabelle, formule, schemi
- **Supporto PDF Ibrido**: Testo nativo + OCR per documenti scansionati
- **Cache OCR**: Evita riprocessamento con sistema di cache intelligente

### 💾 **Ottimizzazione Memoria**
- **Monitoraggio Real-time**: Tracking utilizzo memoria con baseline e picchi
- **Batch Processing Intelligente**: Dimensionamento dinamico basato su memoria disponibile
- **Cache Memory-Aware**: LRU cache con cleanup automatico
- **Garbage Collection Aggressivo**: Strategie GC per documenti di grandi dimensioni

### 🔍 **Strategie di Retrieval Avanzate**
- **Chain-of-Thought**: Ragionamento multi-step per query complesse
- **Multi-Query**: Generazione automatica di strategie di ricerca multiple
- **HyDE**: Hypothetical Document Embeddings per ricerca semantica migliorata
- **Parent-Child Architecture**: Struttura gerarchica per preservazione contesto

### 🎨 **Interfaccia Utente Evoluta**
- **Visualizzatore PDF Integrato**: Navigazione pagine con link diretti alle fonti
- **Configurazione Real-time**: Switch dinamico tra modelli e strategie
- **Modalità Debug**: Strumenti per analizzare retrieval e performance
- **Messaggi di Caricamento Creativi**: Indicatori di progresso a tema ingegneristico

## 🚀 Quick Start

### 1. **Setup Ambiente di Sviluppo**

```bash
# Clone repository
git clone https://github.com/yourusername/AI-syd-Gold.git
cd AI-syd-Gold

# Installa dipendenze complete (include test e sviluppo)
pip install -r requirements-dev.txt
```

### 2. **Configurazione API Keys**

Crea un file `.env` basato su `.env.example`:

```bash
cp .env.example .env
```

Configura le chiavi API necessarie:

```env
# Almeno uno dei seguenti provider
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Configurazione LLM (opzionale)
LLM_PROVIDER=google                    # google, openai, anthropic, ollama
LLM_MODEL_NAME=gemini-2.0-flash-exp   # Modello specifico

# Configurazione Embedding (opzionale)
EMBEDDING_PROVIDER=huggingface         # google, openai, huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Ottimizzazioni Performance (opzionale)
EMBEDDING_TIMEOUT=180
EMBEDDING_RETRY_ATTEMPTS=3
MEMORY_THRESHOLD_MB=2048
```

### 3. **Aggiunta Documenti PDF**

```bash
# Aggiungi i tuoi PDF tecnici
mkdir -p pdfs
cp your_technical_documents.pdf pdfs/
```

### 4. **Costruzione Vector Store**

```bash
# Build completo con elaborazione multimodale
python build_hybrid_store.py

# Alternativa: build rapido (solo testo)
python quick_vector_build.py
```

### 5. **Avvio Applicazione**

```bash
# Interfaccia web Streamlit
streamlit run streamlit_app.py

# CLI diretto (deprecato)
python app.py
```

## 🛠️ Deployment

### **Streamlit Cloud** (Consigliato)

1. **Push su GitHub**:
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy su Streamlit Cloud**:
   - Vai su [share.streamlit.io](https://share.streamlit.io)
   - Collega il repository GitHub
   - Configura le variabili d'ambiente nella sezione "Advanced settings"

3. **Variabili d'Ambiente su Streamlit Cloud**:
```toml
GOOGLE_API_KEY = "your_api_key"
LLM_PROVIDER = "google"
EMBEDDING_PROVIDER = "huggingface"
```

### **Sviluppo Locale con Dev Container**

```bash
# Con VSCode + Dev Containers extension
code .
# Clicca "Reopen in Container" quando richiesto
```

### **GitHub Codespaces**

1. Clicca il bottone "Code" → "Codespaces" → "Create codespace on main"
2. Attendi setup automatico
3. Configura `.env` con le API keys
4. Run: `streamlit run streamlit_app.py`

## 📁 Architettura del Progetto

```text
AI-syd-Gold/
├── 🎯 Core Application
│   ├── app.py                     # Logica applicativa principale
│   ├── streamlit_app.py           # Interfaccia web Streamlit
│   ├── llm_providers.py           # Factory multi-provider AI
│   └── system_prompt.py           # Gestione prompts centralizzata
├── ⚙️ Configuration
│   └── config/
│       ├── __init__.py
│       └── settings.py            # Configurazione centralizzata
├── 🔧 Extensions & Utilities
│   ├── extensions/
│   │   └── multimodal.py         # Elaborazione OCR e immagini
│   └── utils/
│       └── memory_optimizer.py   # Ottimizzazione memoria
├── 🧪 Testing & Quality
│   ├── tests/
│   │   ├── unit/                 # Test unitari
│   │   ├── integration/          # Test integrazione
│   │   ├── performance/          # Test performance
│   │   └── conftest.py           # Configurazione test
│   ├── pyproject.toml            # Configurazione progetto moderna
│   └── requirements-dev.txt      # Dipendenze sviluppo
├── 🏗️ Build Scripts
│   ├── build_hybrid_store.py     # Builder vector store completo
│   ├── build_vector_store.py     # Builder con cache multimodale
│   ├── quick_vector_build.py     # Builder rapido
│   └── build_cache.py            # Pre-build cache
├── 📦 Deployment
│   ├── requirements.txt          # Dipendenze produzione
│   ├── .devcontainer/           # Setup container sviluppo
│   └── .env.example             # Template configurazione
└── 📚 Knowledge Base
    ├── pdfs/                    # Documenti PDF tecnici
    ├── cache/                   # Cache elaborazioni
    └── vector_store_cache/      # Cache embedding
```

## 🔧 Provider Supportati

### **Large Language Models**

| Provider | Modelli | Caratteristiche |
|----------|---------|-----------------|
| **Google** | `gemini-2.0-flash-exp`, `gemini-2.5-flash`, `gemini-2.5-pro` | Veloce, ottimo per RAG, supporto multimodale |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `o1-preview`, `o1-mini` | Qualità premium, ragionamento avanzato |
| **Anthropic** | `claude-3.5-haiku`, `claude-4-sonnet`, `claude-4-opus` | Testo lungo, analisi dettagliate |
| **Ollama** | `llama3.2`, `mistral`, `codellama` | Locale, privacy, nessun costo API |

### **Embedding Models**

| Provider | Modelli | Use Case |
|----------|---------|----------|
| **HuggingFace** | `sentence-transformers/all-MiniLM-L6-v2` | Locale, veloce, gratuito |
| **Google** | `models/embedding-001`, `text-embedding-004` | Ottimizzato per Gemini |
| **OpenAI** | `text-embedding-3-large`, `text-embedding-3-small` | Alta qualità, multilingue |

## 🎛️ Configurazione Avanzata

### **Ottimizzazione Performance**

```env
# Memory Management
MEMORY_THRESHOLD_MB=2048           # Soglia memoria per batch processing
MAX_BATCH_SIZE=50                  # Dimensione massima batch
ENABLE_MEMORY_MONITORING=true     # Abilita monitoraggio memoria

# Retrieval Optimization
RETRIEVER_K=5                      # Numero documenti recuperati
CHUNK_SIZE=1000                    # Dimensione chunk testo
CHUNK_OVERLAP=200                  # Overlap tra chunk

# API Optimization
EMBEDDING_TIMEOUT=180              # Timeout embedding (secondi)
EMBEDDING_RETRY_ATTEMPTS=3         # Tentativi retry
EMBEDDING_RETRY_DELAY=5            # Delay tra retry
```

### **Configurazione Multimodale**

```env
# OCR Settings
ENABLE_OCR=true                    # Abilita elaborazione OCR
OCR_LANGUAGES=ita+eng              # Lingue OCR (Tesseract)
OCR_MIN_CONFIDENCE=60              # Confidenza minima OCR

# Image Processing
ENABLE_IMAGE_PROCESSING=true       # Elaborazione immagini avanzata
IMAGE_DPI=300                      # DPI per rendering PDF
TABLE_DETECTION=true               # Rilevamento tabelle automatico
```

## 📊 Funzionalità di Monitoraggio

### **Logging Strutturato**

Il sistema include logging completo per:

- 📈 **Performance Metrics**: Tempi di risposta, uso memoria, cache hit rate
- 🔍 **Retrieval Analysis**: Documenti trovati, rilevanza, strategie utilizzate
- 🖼️ **Multimodal Processing**: Risultati OCR, classificazione immagini
- 🚨 **Error Tracking**: Errori API, timeout, fallback utilizzati

### **Debug Mode**

```bash
# Abilita logging dettagliato
export LOG_LEVEL=DEBUG
streamlit run streamlit_app.py
```

Nell'interfaccia Streamlit:
- **Analisi Retrieval**: Visualizza documenti recuperati e scoring
- **Ispezione Cache**: Stato cache e statistiche hit/miss
- **Monitor Memoria**: Utilizzo memoria real-time
- **Performance Panel**: Tempi di elaborazione per componente

## 🧪 Testing

### **Esecuzione Test**

```bash
# Test completi con coverage
python run_tests.py --coverage

# Test specifici
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# Test con report HTML
pytest --cov=. --cov-report=html
```

### **Quality Checks**

```bash
# Linting e formatting
black .
flake8 .
mypy .
isort .

# Pre-commit hooks (opzionale)
pre-commit install
pre-commit run --all-files
```

## 🛠️ Risoluzione Problemi

### **Errori Comuni**

#### **"ModuleNotFoundError: No module named 'langchain_huggingface'"**

```bash
pip install langchain-huggingface>=0.0.3 sentence-transformers>=2.2.0
```

#### **"Tesseract not found"**

**Windows**:
```bash
# Download e installa Tesseract da: https://github.com/UB-Mannheim/tesseract/wiki
# Il sistema rileva automaticamente il path
```

**Linux/Mac**:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ita  # Ubuntu
brew install tesseract tesseract-lang                  # macOS
```

#### **"Memory error durante vector store build"**

```env
# Riduci batch size nel .env
MAX_BATCH_SIZE=10
MEMORY_THRESHOLD_MB=1024
```

#### **"API Rate Limit"**

```env
# Aumenta delay tra richieste
EMBEDDING_RETRY_DELAY=10
EMBEDDING_TIMEOUT=300
```

### **Performance Tuning**

#### **Per Documenti di Grandi Dimensioni**

```bash
# Use quick build per test iniziali
python quick_vector_build.py

# Poi upgrading a full build
python build_hybrid_store.py --incremental
```

#### **Ottimizzazione Memoria**

```env
# Configurazione per sistemi con poca RAM
MEMORY_THRESHOLD_MB=512
MAX_BATCH_SIZE=5
ENABLE_AGGRESSIVE_GC=true
```

## 🔗 Integrazione e API

### **Uso come Libreria**

```python
from app import create_chatbot
from config import AppConfig

# Configurazione
config = AppConfig()
config.llm_provider = "google"
config.llm_model_name = "gemini-2.0-flash-exp"

# Creazione chatbot
qa_chain = create_chatbot("standard", config)

# Query
response = qa_chain.invoke({"query": "Come funzionano i cuscinetti a sfera?"})
print(response["result"])
```

### **API REST (Sviluppo Futuro)**

```python
# Pianificato per versioni future
# POST /api/query
# GET /api/health
# GET /api/models
```

## 🤝 Contributi

1. **Fork** il repository
2. **Crea** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** le modifiche (`git commit -m 'Add AmazingFeature'`)
4. **Push** al branch (`git push origin feature/AmazingFeature`)
5. **Apri** Pull Request

### **Standard di Sviluppo**

- ✅ **Type Hints**: Tutte le funzioni devono avere type annotations
- ✅ **Testing**: Coverage minimo 80% per nuove feature
- ✅ **Documentation**: Docstring per classi e funzioni pubbliche
- ✅ **Code Style**: Conformità a Black, flake8, mypy

## 📝 Changelog

### **v2.0** (Corrente)
- ✨ Supporto multi-provider AI (Google, OpenAI, Anthropic, Ollama)
- 🖼️ Elaborazione multimodale con OCR intelligente
- 💾 Sistema ottimizzazione memoria avanzato
- 🎨 Interfaccia Streamlit completamente rinnovata
- 🧪 Framework testing completo
- 📦 Setup deployment moderno

### **v1.0** (Legacy)
- 📄 Chatbot PDF base con Google Gemini
- 🔍 Retrieval semplice con FAISS
- 💬 Interfaccia console basica

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori dettagli.

## 🙏 Riconoscimenti

- **LangChain** - Framework RAG
- **Streamlit** - Interfaccia web
- **FAISS** - Vector database
- **Tesseract** - Engine OCR
- **HuggingFace** - Modelli embedding
- **Google AI** - Gemini models

---

**Sviluppato per l'eccellenza nell'ingegneria meccanica** 🔧⚙️