#!/bin/bash

# 1. Das virtuelle Environment aktivieren (DAS IST NEU UND WICHTIG!)
source /opt/venv/bin/activate

# 2. MLflow im Hintergrund starten
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////workspace/mlflow.db \
    --default-artifact-root /workspace/mlruns &

# 3. Jupyter Lab starten (mit deaktivierten Proxy-Blockaden)
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.disable_check_xsrf=True