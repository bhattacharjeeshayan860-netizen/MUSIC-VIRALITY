from select_features import FEATURE_COLUMNS, TARGET
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def run_training_pipeline():
    x_train, x_test, y_train, y_test = train_test_split(FEATURE_COLUMNS,
                                                        TARGET,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=TARGET,
                                                        )
    model1=RandomForestClassifier(random_state=42,)
    model2=XGBClassifier(random_state=42,)
    model3=LogisticRegression(random_state=42, max_iter=1000)
    print("Training Random Forest...")
    model1.fit(x_train,y_train)
    print("Training XGBoost...")
    model2.fit(x_train,y_train)
    print("Training Logistic Regression...")
    model3.fit(x_train,y_train)
    return 
