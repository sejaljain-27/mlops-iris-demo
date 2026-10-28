pipeline {
    agent any

    environment {
        APP_DIR = '/home/ubuntu/app'
        VENV_DIR = '.venv'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set up Python') {
            steps {
                sh '''
                    set -eu
                    python3 --version
                    python3 -m venv "$VENV_DIR"
                    "$VENV_DIR/bin/python" -m pip install --upgrade pip
                    "$VENV_DIR/bin/pip" install -r requirements.txt
                '''
            }
        }

        stage('Train / Retrain Model') {
            steps {
                sh '''
                    set -eu
                    "$VENV_DIR/bin/python" retrain.py
                    test -s iris_model.pkl
                '''
            }
        }

        stage('Smoke Test') {
            steps {
                sh '''
                    set -eu
                    "$VENV_DIR/bin/python" - <<'PY'
import joblib
import pandas as pd
model = joblib.load("iris_model.pkl")
features = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
prediction = model.predict(features)
assert len(prediction) == 1
print("Model artifact loaded successfully; sample prediction:", prediction[0])
PY
                '''
            }
        }

        stage('Deploy Model Artifact') {
            steps {
                sh '''
                    set -eu
                    if [ ! -d "$APP_DIR" ]; then
                        echo "Deployment directory $APP_DIR does not exist."
                        echo "Create it and grant the Jenkins service account write access before deploying."
                        exit 1
                    fi
                    install -m 0644 iris_model.pkl "$APP_DIR/iris_model.pkl"
                    echo "Model artifact copied to $APP_DIR/iris_model.pkl"
                '''
            }
        }
    }

    post {
        success {
            echo 'MLOps pipeline completed successfully.'
        }
        failure {
            echo 'MLOps pipeline failed. Review the stage logs above.'
        }
    }
}
