#!/bin/bash

# 1. MLflow im Hintergrund starten (das Kaufmanns-Und '&' ist wichtig!)
# Wir legen die Datenbank und die Artefakte (Modelle/Plots) zwingend in 
# /workspace ab. Nur so überleben sie einen Pod-Neustart!
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////workspace/mlflow.db \
    --default-artifact-root /workspace/mlruns &

# 2. Jupyter Lab im Vordergrund starten
# Dies hält den Container dauerhaft am Leben.
jupyter lab \
    --ip='*' \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token=''