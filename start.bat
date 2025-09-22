@echo off
REM =================================================================
REM AI-syd-Gold Startup Script
REM Gestisce l'avvio, riavvio e chiusura dell'applicazione Streamlit
REM =================================================================

REM 1. Assicurati di aver installato le dipendenze:
REM    pip install -r requirements.txt
REM 2. Configura la chiave API Google in .env (vedi README.md)
REM 3. Inserisci i PDF tecnici nella cartella pdfs\
REM 4. Crea la cache degli embedding:
REM    python build_cache.py
REM 5. Avvia il chatbot Streamlit:
py -m streamlit run streamlit_app.py
