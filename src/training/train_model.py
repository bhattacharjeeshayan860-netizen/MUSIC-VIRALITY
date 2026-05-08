from src.features.build_features import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

def run_training_pipeline():
    X, y = get_X_y()
    if X.empty or y.empty:
        print("No labeled data available for training.")
        return

    stratify = y if y.nunique(dropna=False) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

#Random forest did well but the got overfitted as it made no mistakes on the test set.
#logistic regression did way too bad, so I removed it from the final model selection

    model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            random_state=42,
            eval_metric="logloss",
        )


    print(f"Training XGBoost...")
    model.fit(x_train, y_train)

    print("Training complete.")
    print(f"Evaluation for XGBoost:")
    preds = model.predict(x_test)
    print("Confusion Matrix: \n")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report: \n")
    print(classification_report(y_test, preds))
    joblib.dump(model, "models/trained/model.pkl")
    print("Model saved to models/trained/model.pkl")

    return
