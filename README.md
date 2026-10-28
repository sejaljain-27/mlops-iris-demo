# MLOps Iris Demo — Jenkins Pipeline

A demo pipeline that retrains an Iris classifier with PyCaret, smoke-tests the generated model, and copies `iris_model.pkl` to an application directory.

## Files

- `app.py`: Flask prediction API
- `train_model.py`: baseline model training script
- `retrain.py`: PyCaret retraining script used by Jenkins
- `requirements.txt`: Python dependencies
- `Jenkinsfile`: declarative Jenkins pipeline

## Run locally

Use Python 3.10–3.12, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python retrain.py
python app.py
```

Health check: `curl http://127.0.0.1:5000/health`

Prediction request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Jenkins setup

1. Create a Pipeline job using **Pipeline script from SCM** and the repository's Git URL.
2. Ensure the Jenkins agent has Python 3 and the `python3-venv` OS package installed.
3. The demo deploys to `/home/ubuntu/app`. Create this directory and grant the Jenkins service account the required write permission before building.
4. If Jenkins and the Flask app run on different VMs, a local `install` command deploys only on the Jenkins agent. Use secure SSH-based deployment for a separate target VM.
5. Restrict access and secure the Flask app before exposing it to a network.

A starter `iris_model.pkl` is included so the Flask API can run immediately after dependencies are installed. Jenkins retrains and replaces this file during every successful pipeline run; for production repositories, consider storing model artifacts in an artifact registry or object storage instead of Git.
