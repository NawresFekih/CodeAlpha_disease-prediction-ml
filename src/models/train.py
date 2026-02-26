from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

from src.data.preprocess import load_and_preprocess
from src.utils.seed import set_seed


def train():

    set_seed(42)

    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")

    joblib.dump(
        model,
        "outputs/saved_models/xgboost_model.pkl"
    )


if __name__ == "__main__":
    train()