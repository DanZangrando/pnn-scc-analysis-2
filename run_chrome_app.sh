#!/bin/bash

# Navegar al directorio del proyecto
PROJECT_DIR="/home/histolab/Proyectos/pnn-scc-analysis-2"
cd "$PROJECT_DIR"

PORT=8501

# Verificar si el puerto ya está en uso (Streamlit ya está corriendo)
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    # Ya está corriendo, simplemente abrimos Chrome en modo aplicación
    google-chrome --app="http://localhost:$PORT"
else
    # No está corriendo, iniciamos Streamlit en segundo plano usando uv
    uv run streamlit run app.py --server.port $PORT --server.headless true &
    STREAMLIT_PID=$!
    
    # Esperamos a que el servidor de Streamlit se inicie (máximo 5 segundos)
    for i in {1..10}; do
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            break
        fi
        sleep 0.5
    done
    
    # Abrimos Chrome en modo aplicación (sin barras de navegación, como app nativa)
    google-chrome --app="http://localhost:$PORT"
    
    # Al cerrar la ventana de Chrome, detenemos el proceso de Streamlit
    kill $STREAMLIT_PID 2>/dev/null
fi
