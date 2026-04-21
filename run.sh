#!/bin/bash
# run.sh — Start all AdvisorIQ services
# Usage: bash run.sh [streamlit|api|scheduler|all]

cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment loaded from .env"
fi

MODE=${1:-all}

start_streamlit() {
    echo "▶ Starting Streamlit UI on http://localhost:8501"
    streamlit run app.py --server.port 8501 --server.headless true &
    STREAMLIT_PID=$!
    echo "  Streamlit PID: $STREAMLIT_PID"
}

start_api() {
    echo "▶ Starting FastAPI on http://localhost:8000"
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    echo "  FastAPI PID: $API_PID"
}

start_scheduler() {
    echo "▶ Starting background scheduler"
    python scheduler.py &
    SCHED_PID=$!
    echo "  Scheduler PID: $SCHED_PID"
}

train_models() {
    echo "▶ Training ML models (first run)..."
    python ml_model.py
    echo "✓ Models ready"
}

init_db() {
    echo "▶ Initialising database..."
    python -c "import database; database.init_db(); print('✓ Database ready')"
}

case $MODE in
    streamlit)
        init_db
        train_models
        start_streamlit
        wait
        ;;
    api)
        init_db
        train_models
        start_api
        wait
        ;;
    scheduler)
        start_scheduler
        wait
        ;;
    all)
        init_db
        train_models
        start_streamlit
        start_api
        start_scheduler
        echo ""
        echo "═══════════════════════════════════════"
        echo "  AdvisorIQ running:"
        echo "  UI:        http://localhost:8501"
        echo "  API:       http://localhost:8000"
        echo "  API Docs:  http://localhost:8000/docs"
        echo "  Press Ctrl+C to stop all"
        echo "═══════════════════════════════════════"
        wait
        ;;
    *)
        echo "Usage: bash run.sh [streamlit|api|scheduler|all]"
        exit 1
        ;;
esac
