#!/bin/bash
PROJECT_DIR="/home/histolab/Proyectos/pnn-scc-analysis-2"
cd "$PROJECT_DIR" || exit 1

LAUNCHER_LOG="$PROJECT_DIR/launcher.log"
echo "--- Launcher started at $(date) ---" >> "$LAUNCHER_LOG"

export PATH="$PROJECT_DIR/.venv/bin:/home/histolab/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

PORT=8501
CHROME_BIN="/usr/bin/google-chrome"

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Streamlit ya está corriendo en el puerto $PORT. Abriendo Chrome..." >> "$LAUNCHER_LOG"
    "$CHROME_BIN" --app="http://localhost:$PORT" &
else
    echo "Iniciando Streamlit..." >> "$LAUNCHER_LOG"
    nohup streamlit run app.py --server.port $PORT --server.headless true > "$PROJECT_DIR/streamlit.log" 2>&1 &
    STREAMLIT_PID=$!
    disown $STREAMLIT_PID
    
    for i in {1..20}; do
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            break
        fi
        sleep 0.5
    done
    
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo "Streamlit iniciado correctamente. Abriendo Google Chrome..." >> "$LAUNCHER_LOG"
        "$CHROME_BIN" --app="http://localhost:$PORT" &
    else
        echo "Error: No se pudo detectar Streamlit en el puerto $PORT." >> "$LAUNCHER_LOG"
        tail -n 20 "$PROJECT_DIR/streamlit.log" >> "$LAUNCHER_LOG"
        exit 1
    fi
fi
