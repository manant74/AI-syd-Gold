# AI-syd-Gold - Chatbot PDF per Ingegneria Meccanica

Un chatbot intelligente specializzato in ingegneria dei materiali e meccanica, con focus sui cuscinetti, che utilizza documenti PDF tecnici per fornire risposte precise.

## 🚀 Setup Rapido

### 1. Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### 2. Configurazione API Google
1. Ottieni una chiave API da [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea un file `.env` basato su `env_template.txt`:
```bash
cp env_template.txt .env
```
3. Modifica il file `.env` inserendo la tua chiave API:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Aggiunta Documenti PDF
Posiziona i tuoi documenti PDF tecnici nella cartella `pdfs/`.

### 4. Creazione Cache
```bash
python build_cache.py
```

### 5. Avvio Chatbot
```bash
python app.py
```

## 🔧 Risoluzione Problemi

### Errore di Timeout durante la Creazione Cache

Se incontri errori come:
```
Error embedding content: Timeout of 60.0s exceeded, last exception: 503 IOCP/Socket: Connection reset
```

#### Soluzioni:

1. **Verifica Connessione Internet**
   - Assicurati di avere una connessione stabile
   - Prova a disabilitare temporaneamente firewall/proxy

2. **Configurazione API**
   - Verifica che la chiave API sia corretta nel file `.env`
   - Controlla che la chiave abbia i permessi per l'API Gemini
   - Verifica che la chiave non sia scaduta

3. **Aumenta Timeout**
   Modifica il file `.env`:
   ```
   EMBEDDING_TIMEOUT=180
   EMBEDDING_RETRY_ATTEMPTS=5
   EMBEDDING_RETRY_DELAY=10
   ```

4. **Problemi di Rete**
   - Se usi un proxy aziendale, configura le variabili d'ambiente:
     ```bash
     export HTTP_PROXY=http://proxy.company.com:8080
     export HTTPS_PROXY=http://proxy.company.com:8080
     ```

5. **Alternative per Ambienti Limitati**
   - Usa una connessione internet diversa
   - Prova in orari di minor traffico
   - Considera l'uso di un VPN

### Altri Errori Comuni

#### "File .env non trovato"
```bash
cp env_template.txt .env
# Poi modifica .env con la tua chiave API
```

#### "Nessun file PDF trovato"
- Assicurati che i PDF siano nella cartella `pdfs/`
- Verifica che i PDF non siano protetti da password
- Controlla che i PDF contengano testo (non solo immagini)

#### "Errore nel caricamento dei documenti"
```bash
pip install unstructured pdfplumber pymupdf
```

## 📁 Struttura Progetto

```
AI-syd-Gold/
├── app.py                 # Chatbot principale
├── build_cache.py         # Script per creare la cache
├── streamlit_app.py       # Interfaccia web Streamlit
├── requirements.txt       # Dipendenze Python
├── env_template.txt       # Template configurazione
├── pdfs/                 # Documenti PDF
├── vector_store_cache/    # Cache degli embedding
└── README.md             # Questo file
```

## 🎯 Funzionalità

- **Recupero Intelligente**: Utilizza ParentDocumentRetriever per migliorare il contesto
- **Multiple Strategie**: Supporta ensemble, HyDE, e multi-query retrieval
- **Cache Intelligente**: Salva e riutilizza gli embedding per performance ottimali
- **Validazione PDF**: Verifica automatica della qualità dei documenti
- **Debug Avanzato**: Strumenti per analizzare il contenuto e il recupero

## 🔍 Modalità di Recupero

1. **Standard**: ParentDocumentRetriever con cache
2. **Ensemble**: Combina BM25 e FAISS per risultati migliori
3. **HyDE**: Hypothetical Document Embedding per query più precise
4. **Multi-Query**: Genera multiple query per migliorare il recupero

## 📊 Monitoraggio

Il sistema include logging dettagliato per:
- Caricamento documenti
- Creazione embedding
- Processo di cache
- Performance del retriever

## 🤝 Contributi

Per segnalare problemi o suggerire miglioramenti, apri una issue su GitHub.