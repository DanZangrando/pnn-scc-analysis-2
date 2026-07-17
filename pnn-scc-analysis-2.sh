#!/bin/bash

# Ruta al proyecto
PROJECT_DIR="/home/histolab/Proyectos/pnn-scc-analysis-2"
cd "$PROJECT_DIR" || exit 1

# Archivo de log del launcher
LAUNCHER_LOG="$PROJECT_DIR/launcher.log"
echo "--- Launcher started at $(date) ---" >> "$LAUNCHER_LOG"

# Asegurar que las rutas del sistema y del entorno virtual están en el PATH
export PATH="$PROJECT_DIR/.venv/bin:/home/histolab/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Activar el entorno virtual del proyecto
source .venv/bin/activate

PORT=8501
CHROME_BIN="/usr/bin/google-chrome"

# Verificar si Streamlit ya está corriendo en el puerto 8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Streamlit ya está corriendo en el puerto $PORT. Abriendo Chrome..." >> "$LAUNCHER_LOG"
    "$CHROME_BIN" --app="http://localhost:$PORT" &
else
    echo "Iniciando Streamlit..." >> "$LAUNCHER_LOG"
    # Ejecutamos streamlit en segundo plano desvinculado con nohup, guardando logs
    nohup streamlit run app.py --server.port $PORT --server.headless true > "$PROJECT_DIR/streamlit.log" 2>&1 &
    STREAMLIT_PID=$!
    
    # Desasociar del shell para evitar que muera al cerrar el script
    disown $STREAMLIT_PID
    
    # Esperar hasta 10 segundos a que Streamlit levante el servidor
    for i in {1..20}; do
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            break
        fi
        sleep 0.5
    done
    
    # Verificar si el servidor se inició correctamente
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo "Streamlit iniciado correctamente. Abriendo Google Chrome..." >> "$LAUNCHER_LOG"
        "$CHROME_BIN" --app="http://localhost:$PORT" &
    else
        echo "Error: No se pudo detectar Streamlit en el puerto $PORT." >> "$LAUNCHER_LOG"
        echo "Últimas líneas del log de Streamlit:" >> "$LAUNCHER_LOG"
        tail -n 20 "$PROJECT_DIR/streamlit.log" >> "$LAUNCHER_LOG"
        exit 1
    fi
fi
