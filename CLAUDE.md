# CLAUDE.md - Analisi e Ottimizzazioni AI-syd-Gold

## Panoramica del Progetto

**AI-syd-Gold** è un chatbot PDF specializzato in ingegneria meccanica con focus sui cuscinetti, che utilizza tecnologie RAG (Retrieval-Augmented Generation) per fornire risposte tecniche precise basate su documenti PDF.

### Architettura Attuale

- **Backend**: Python con LangChain e Google Generative AI
- **Frontend**: Streamlit per interfaccia web
- **Storage**: FAISS per vector store, cache locale con pickle
- **Retrieval**: ParentDocumentRetriever con multiple strategie
- **Knowledge Base**: 19 documenti PDF tecnici sui cuscinetti

## 🔧 Comando per Lint/Typecheck

```bash
# Al momento non definiti nel progetto
# Suggeriti:
python -m flake8 --max-line-length=120 *.py
python -m mypy *.py --ignore-missing-imports
```

## 📊 Analisi del Codice

### Punti di Forza

1. **Architettura RAG robusta** con ParentDocumentRetriever
2. **Multiple strategie di retrieval** (standard, HyDE, multi-query, ensemble)
3. **Sistema di cache intelligente** con validazione MD5
4. **Gestione errori estensiva** con retry logic e timeout configurabili
5. **Debug capabilities** integrate per analisi del retrieval
6. **Interfaccia utente intuitiva** con Streamlit

### Criticità Identificate

1. **Codice duplicato** nel prompt template (app.py:48-88 e 502-542)
2. **Gestione memoria** inefficiente per documenti molto grandi
3. **Mancanza di configurazione** per modelli embedding alternativi
4. **Logging non strutturato** difficile da analizzare in produzione
5. **Sicurezza** chiave API in .env senza validazione robusta

## 🚀 Proposte di Ottimizzazione

### 1. Ottimizzazioni delle Performance

#### Gestione Memoria Migliorata

```python
# Aggiungere in app.py
import gc
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_embed_query(query_text):
    """Cache embedding queries per evitare ricalcoli."""
    return embeddings.embed_query(query_text)

def process_documents_in_chunks(documents, chunk_size=10):
    """Processa documenti in chunk per ridurre utilizzo memoria."""
    for i in range(0, len(documents), chunk_size):
        yield documents[i:i + chunk_size]
        gc.collect()  # Forza garbage collection
```

#### Ottimizzazione Vector Store

```python
# Nuovo file: optimized_retriever.py
class OptimizedFAISSRetriever:
    def __init__(self, faiss_index_path, embedding_model):
        self.index = faiss.read_index(faiss_index_path)
        self.embedding_model = embedding_model
        self._query_cache = {}

    def similarity_search_with_cache(self, query, k=5):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self._query_cache:
            return self._query_cache[query_hash]

        results = self._perform_search(query, k)
        self._query_cache[query_hash] = results
        return results
```

### 2. Refactoring del Codice

#### Separazione delle Responsabilità

```python
# config/settings.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    llm_model: str = "gemini-1.5-flash"
    retriever_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_timeout: int = 60

    @classmethod
    def from_env(cls):
        return cls(
            llm_model=os.getenv("LLM_MODEL_NAME", cls.llm_model),
            retriever_k=int(os.getenv("RETRIEVER_K", cls.retriever_k)),
            # ... altri parametri
        )

# prompts/templates.py
class PromptTemplates:
    EXPERT_ASSISTANT = """
    Sei BearX, un assistente tecnico specializzato in ingegneria meccanica...
    {context}
    {question}
    """

    HYDE_TEMPLATE = """
    Sei un assistente utile. Il tuo compito è generare...
    Domanda: {question}
    Paragrafo di risposta ipotetico:
    """
```

### 3. Monitoring e Observability

#### Logging Strutturato

```python
# utils/logger.py
import structlog
import json

def setup_structured_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Utilizzo:
logger = structlog.get_logger()
logger.info("query_processed",
           query=user_question,
           retrieved_docs_count=len(docs),
           processing_time=elapsed_time)
```

## 🌟 Proposte di Estensione Funzionalità

### 1. Sistema Multi-Modale

#### Supporto per Immagini e Diagrammi

```python
# extensions/multimodal.py
from langchain.document_loaders import UnstructuredImageLoader
import pytesseract
from PIL import Image

class MultimodalDocumentProcessor:
    def __init__(self):
        self.image_loader = UnstructuredImageLoader()

    def extract_text_from_images(self, pdf_path):
        """Estrae testo da immagini nei PDF usando OCR."""
        images = self._extract_images_from_pdf(pdf_path)
        extracted_texts = []

        for img in images:
            text = pytesseract.image_to_string(img, lang='ita+eng')
            extracted_texts.append(text)

        return extracted_texts

    def process_technical_diagrams(self, image):
        """Identifica e elabora diagrammi tecnici."""
        # Integrazione con modelli di visione per analisi diagrammi
        pass
```

### 2. Sistema di Validazione e Quality Assurance

#### Validation Framework

```python
# validation/qa_system.py
class DocumentQualityValidator:
    def __init__(self):
        self.quality_metrics = {
            'text_density': 0.1,  # Minimo rapporto testo/pagina
            'technical_terms': ['bearing', 'cuscinetto', 'lubrificazione'],
            'formula_patterns': [r'\w+\s*=\s*\w+', r'\d+\.\d+']
        }

    def validate_document_quality(self, document):
        scores = {}
        scores['text_density'] = self._calculate_text_density(document)
        scores['technical_relevance'] = self._check_technical_terms(document)
        scores['formula_presence'] = self._detect_formulas(document)

        return scores

    def suggest_improvements(self, scores):
        suggestions = []
        if scores['text_density'] < self.quality_metrics['text_density']:
            suggestions.append("Documento potrebbe essere principalmente immagini - considera OCR")
        return suggestions
```

### 3. Sistema di Feedback e Apprendimento

#### User Feedback Loop

```python
# feedback/learning.py
class FeedbackSystem:
    def __init__(self, db_path="feedback.db"):
        self.db_path = db_path
        self._init_database()

    def record_interaction(self, query, response, user_rating, sources_used):
        """Registra interazione utente per miglioramento continuo."""
        interaction = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'rating': user_rating,
            'sources': sources_used,
            'retriever_type': self.current_retriever_type
        }
        self._save_to_db(interaction)

    def analyze_performance_patterns(self):
        """Analizza pattern di performance per ottimizzazioni."""
        interactions = self._load_interactions()

        # Analisi per tipo di query
        query_performance = {}
        for interaction in interactions:
            query_type = self._classify_query(interaction['query'])
            if query_type not in query_performance:
                query_performance[query_type] = []
            query_performance[query_type].append(interaction['rating'])

        return query_performance
```

### 4. API REST e Integrazione

#### REST API Service

```python
# api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="BearX API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    retriever_type: str = "standard"
    max_docs: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence_score: float
    processing_time: float

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()

    try:
        qa_chain = get_cached_chatbot(request.retriever_type)
        response = qa_chain.invoke({"query": request.question})

        processing_time = time.time() - start_time

        return QueryResponse(
            answer=response["result"],
            sources=[doc.metadata.get('source') for doc in response["source_documents"]],
            confidence_score=calculate_confidence(response),
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5. Sistema di Deployment e Scaling

#### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build cache per ridurre startup time
RUN python build_cache.py

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bearx-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bearx-chatbot
  template:
    metadata:
      labels:
        app: bearx-chatbot
    spec:
      containers:
      - name: bearx
        image: bearx-chatbot:latest
        ports:
        - containerPort: 8501
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: google-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
```

## 📋 Piano di Implementazione Prioritizzato

### Fase 1: Ottimizzazioni Core (1-2 settimane)

1. Refactoring configurazione e prompts
2. Implementazione logging strutturato
3. Ottimizzazione gestione memoria
4. Aggiunta lint/typecheck tools

### Fase 2: Qualità e Monitoring (2-3 settimane)

1. Sistema di validazione documenti
2. Metriche di performance
3. Dashboard di monitoring
4. Test automatizzati

### Fase 3: Funzionalità Avanzate (3-4 settimane)

1. Supporto multi-modale
2. Sistema di feedback
3. API REST
4. Miglioramenti UI/UX

### Fase 4: Deployment e Scaling (2-3 settimane)

1. Containerizzazione
2. CI/CD pipeline
3. Deployment cloud
4. Load balancing

## 🔧 Suggerimenti Implementativi Immediati

### File da creare subito

1. `config/settings.py` - Centralizzazione configurazione
2. `utils/logger.py` - Logging strutturato
3. `tests/test_retrieval.py` - Test automatizzati
4. `requirements-dev.txt` - Dipendenze di sviluppo
5. `pyproject.toml` - Configurazione linting

### Modifiche immediate consigliate

1. Estrarre prompt templates in file separato
2. Aggiungere type hints a tutte le funzioni
3. Implementare exception handling più specifico
4. Creare classe di configurazione centralizzata

## 📚 Risorse e Dipendenze Aggiuntive

```txt
# requirements-dev.txt
black==23.7.0
flake8==6.0.0
mypy==1.5.1
pytest==7.4.0
structlog==23.1.0
fastapi==0.103.0
uvicorn==0.23.2
pytesseract==0.3.10
Pillow==10.0.0
```

Questo documento fornisce una roadmap completa per l'evoluzione del progetto AI-syd-Gold, con focus su scalabilità, manutenibilità e funzionalità avanzate.
