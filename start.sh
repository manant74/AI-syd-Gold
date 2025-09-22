#!/bin/bash
# =================================================================
# AI-syd-Gold Startup Script (Linux/macOS)
# Gestisce l'avvio, riavvio e chiusura dell'applicazione Streamlit
# =================================================================

set -e  # Exit on error

# Configurazione
APP_NAME="AI-syd-Gold"
STREAMLIT_FILE="streamlit_app.py"
PYTHON_EXE="python3"
PORT=8501
LOG_FILE="app.log"
PID_FILE=".streamlit.pid"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo
echo "========================================"
echo "   $APP_NAME - Startup Manager"
echo "========================================"
echo

# Funzione per stampare messaggi colorati
print_status() {
    case $1 in
        "error")   echo -e "${RED}❌ $2${NC}" ;;
        "success") echo -e "${GREEN}✅ $2${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "info")    echo -e "${BLUE}🔍 $2${NC}" ;;
        "rocket")  echo -e "${GREEN}🚀 $2${NC}" ;;
    esac
}

# Verifica che Python sia installato
if ! command -v $PYTHON_EXE &> /dev/null; then
    print_status "error" "Python non trovato nel PATH"
    echo "   Installa Python3 o aggiorna il PATH"
    exit 1
fi

# Verifica che il file Streamlit esista
if [[ ! -f "$STREAMLIT_FILE" ]]; then
    print_status "error" "File $STREAMLIT_FILE non trovato"
    echo "   Assicurati di essere nella directory corretta del progetto"
    exit 1
fi

# Funzione per trovare e terminare processi Streamlit esistenti
cleanup_existing_instances() {
    print_status "info" "Verifica istanze esistenti di Streamlit..."

    # Cerca processi Streamlit
    STREAMLIT_PIDS=$(pgrep -f "streamlit.*run.*$STREAMLIT_FILE" 2>/dev/null || true)

    if [[ -n "$STREAMLIT_PIDS" ]]; then
        print_status "warning" "Trovate istanze Streamlit esistenti"
        echo "$STREAMLIT_PIDS" | while read -r pid; do
            echo "   Terminazione processo PID: $pid"
            kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 2  # Attendi terminazione
        print_status "success" "Istanze esistenti terminate"
    fi

    # Verifica se la porta è ancora in uso
    if lsof -i :$PORT &> /dev/null; then
        print_status "warning" "Porta $PORT ancora in uso, tentativo di liberarla..."

        # Trova e termina processo che usa la porta
        PORT_PID=$(lsof -ti :$PORT 2>/dev/null || true)
        if [[ -n "$PORT_PID" ]]; then
            echo "   Terminazione processo sulla porta $PORT (PID: $PORT_PID)"
            kill -TERM "$PORT_PID" 2>/dev/null || kill -KILL "$PORT_PID" 2>/dev/null || true
            sleep 2
        fi
    fi

    # Pulisci file PID se esiste
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi
}

# Verifica dipendenze Python
check_dependencies() {
    print_status "info" "Verifica dipendenze Python..."

    if ! $PYTHON_EXE -c "import streamlit" 2>/dev/null; then
        print_status "error" "Streamlit non installato"
        echo "   Installa le dipendenze con: pip install -r requirements.txt"
        exit 1
    fi

    # Verifica altre dipendenze critiche
    local deps=("langchain" "google.generativeai" "faiss")
    for dep in "${deps[@]}"; do
        if ! $PYTHON_EXE -c "import $dep" 2>/dev/null; then
            print_status "warning" "Dipendenza $dep non trovata"
            echo "   Potrebbe essere necessario installare dipendenze aggiuntive"
        fi
    done
}

# Verifica configurazione
check_configuration() {
    # Verifica file .env
    if [[ ! -f ".env" ]]; then
        print_status "warning" "File .env non trovato"
        if [[ -f "env_template.txt" ]]; then
            echo "   Creazione da template..."
            cp "env_template.txt" ".env"
            print_status "success" "File .env creato da template"
            print_status "warning" "RICORDA: Configura GOOGLE_API_KEY nel file .env"
        else
            print_status "error" "Template .env non trovato"
            echo "   Crea manualmente il file .env con GOOGLE_API_KEY"
        fi
    fi

    # Verifica directory PDF
    if [[ ! -d "pdfs" ]]; then
        print_status "warning" "Directory 'pdfs' non trovata, creazione..."
        mkdir -p pdfs
        print_status "success" "Directory 'pdfs' creata"
        print_status "warning" "RICORDA: Aggiungi i tuoi documenti PDF nella cartella 'pdfs'"
    fi

    # Conta file PDF
    local pdf_count=$(find pdfs -name "*.pdf" 2>/dev/null | wc -l)

    if [[ $pdf_count -eq 0 ]]; then
        print_status "warning" "Nessun file PDF trovato nella directory 'pdfs'"
        echo "   L'applicazione funzionerà ma senza documenti da processare"
    else
        print_status "success" "Trovati $pdf_count file PDF nella directory 'pdfs'"
    fi

    # Verifica cache
    if [[ -d "vector_store_cache" ]]; then
        print_status "success" "Cache vector store esistente trovata"
    else
        print_status "warning" "Cache vector store non trovata"
        echo "   Verrà creata automaticamente al primo avvio (può richiedere tempo)"
    fi
}

# Funzione per gestire la terminazione
cleanup_on_exit() {
    print_status "info" "Terminazione applicazione..."
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null || true)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
    echo "$(date) - Applicazione terminata" >> "$LOG_FILE"
    print_status "success" "Applicazione terminata"
}

# Gestione segnali per cleanup
trap cleanup_on_exit EXIT INT TERM

# Main execution
main() {
    cleanup_existing_instances
    check_dependencies
    check_configuration

    echo
    print_status "rocket" "Avvio applicazione $APP_NAME..."
    echo "   URL: http://localhost:$PORT"
    echo "   File: $STREAMLIT_FILE"
    echo "   Log: $LOG_FILE"
    echo

    # Log avvio
    echo "$(date) - Avvio applicazione" >> "$LOG_FILE"

    # Avvia Streamlit in background e salva PID
    $PYTHON_EXE -m streamlit run "$STREAMLIT_FILE" \
        --server.port=$PORT \
        --server.address=localhost \
        --server.headless=true \
        --server.runOnSave=true \
        --server.allowRunOnSave=true \
        >> "$LOG_FILE" 2>&1 &

    local streamlit_pid=$!
    echo $streamlit_pid > "$PID_FILE"

    print_status "success" "Applicazione avviata (PID: $streamlit_pid)"
    print_status "info" "Premi Ctrl+C per terminare l'applicazione"

    # Verifica che il processo sia ancora attivo
    sleep 3
    if ! kill -0 $streamlit_pid 2>/dev/null; then
        print_status "error" "Applicazione terminata inaspettatamente"
        echo "   Controlla il file $LOG_FILE per dettagli"
        echo "   Verifica:"
        echo "   - File .env configurato correttamente"
        echo "   - GOOGLE_API_KEY valida"
        echo "   - Connessione internet attiva"
        echo "   - Dipendenze Python installate"
        exit 1
    fi

    # Attendi terminazione del processo
    wait $streamlit_pid
}

# Verifica se lo script è eseguito direttamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi