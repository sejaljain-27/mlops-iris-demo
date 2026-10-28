"""Train a baseline Iris classifier and save iris_model.pkl."""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def main():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "iris_model.pkl")
    print(f"Baseline model saved to iris_model.pkl; test accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
