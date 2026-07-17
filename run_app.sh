#!/bin/bash
# Activar el entorno virtual e iniciar Streamlit
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
source .venv/bin/activate
streamlit run app.py
