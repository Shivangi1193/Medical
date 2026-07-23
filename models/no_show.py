import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def train_classifier(df):
    X = df[["age","gender","DaysWaiting","Scholarship","Hipertension","Diabetes","Alcoholism","Handcap","chronic_conditions"]]
    y = df["no_show"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    f1 = f1_score(y_test, clf.predict(X_test))
    roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

    return clf, f1, roc
