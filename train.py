"""Train a small classifier on the Palmer Penguins dataset and save it.

Run:  python train.py
Output: model/penguin_model.joblib
"""

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

##Import the Palmer Penguins Dataset:
from palmerpenguins import load_penguins
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
##

MODEL_PATH = "model/penguin_model.joblib"


def main():
    penguins_data, penguins_target = load_penguins(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
            penguins_data, penguins_target, test_size=0.2, random_state=42
    )
    model = make_pipeline(IterativeImputer(max_iter=10, random_state=0), DecisionTreeClassifier())
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"Test accuracy: {accuracy:.3f}")

    # Save the class names alongside the model.
    joblib.dump({"model": model, "class_names": list(penguins_target)}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
