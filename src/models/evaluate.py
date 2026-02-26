import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from src.data.preprocess import load_and_preprocess
from src.utils.seed import set_seed

def evaluate():
    set_seed(42)

    # Charger les données
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Charger le modèle
    model = joblib.load("outputs/saved_models/xgboost_model.pkl")

    # Prédictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]  # Pour ROC

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("outputs/figures/confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("outputs/figures/roc_curve.png")
    plt.close()

    print(f"Confusion matrix and ROC curve saved in outputs/figures/ (AUC = {roc_auc:.2f})")

if __name__ == "__main__":
    evaluate()