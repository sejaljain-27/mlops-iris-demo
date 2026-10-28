"""Retrain an Iris classifier with PyCaret and save iris_model.pkl."""
from pycaret.classification import setup, compare_models, finalize_model, save_model
from pycaret.datasets import get_data


def main():
    iris = get_data("iris")
    setup(
        data=iris,
        target="species",
        session_id=42,
        html=False,
        verbose=False,
    )
    best_model = compare_models()
    final_model = finalize_model(best_model)
    save_model(final_model, "iris_model")  # writes iris_model.pkl
    print("Retraining complete. Model artifact saved: iris_model.pkl")


if __name__ == "__main__":
    main()
