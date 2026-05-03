from src.training.select_features import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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

    model1 = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model3 = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    models = {"Random Forest": model1, "Logistic Regression": model3}
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            random_state=42,
            eval_metric="logloss",
        )
    except ModuleNotFoundError:
        print("xgboost is not installed; skipping XGBoost model.")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)

    print("Training complete.")
    for name, model in models.items():
        print(f"Evaluation for {name}:")
        preds = model.predict(x_test)
        print("Confusion Matrix: \n")
        print(confusion_matrix(y_test, preds))
        print("\nClassification Report: \n")
        print(classification_report(y_test, preds))

    return
