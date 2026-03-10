#!/bin/bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/diabetes_classifier/3" -p 8000 --no-conda