# Project Overview

This project appears to be an AI chatbot application, primarily built with Python. It leverages Streamlit for the user interface, allowing for interactive chat experiences. The backend likely handles interactions with various Large Language Model (LLM) providers, as suggested by `llm_providers.py` and the example environment files (`example_anthropic.env`, `example_openai.env`). The project also includes functionality for document processing and vector store management, indicated by the `pdfs` directory and `vector_store_cache`.

# Building and Running

## Prerequisites

*   Python 3.x
*   `pip` (Python package installer)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd AI-syd-Gold
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```
4.  **Configure LLM API Keys:**
    Copy either `example_anthropic.env` or `example_openai.env` to `.env` and fill in your API keys.
    ```bash
    cp example_openai.env .env
    # Or
    cp example_anthropic.env .env
    ```
    Then, edit the `.env` file with your actual API keys.

## Running the Application

The application can be started using Streamlit.

```bash
streamlit run streamlit_app.py
```

Alternatively, you can use the provided start scripts:

*   **Windows:**
    ```bash
    start.bat
    ```
*   **macOS/Linux:**
    ```bash
    ./start.sh
    ```

## Testing

Tests are located in the `tests/` directory. You can run them using `pytest`:

```bash
pytest
```

# Development Conventions

*   **Language:** Python
*   **Dependency Management:** `requirements.txt` and `pyproject.toml` are used for managing project dependencies.
*   **Code Structure:** The project follows a modular structure with directories for configuration (`config/`), utilities (`utils/`), and tests (`tests/`).
