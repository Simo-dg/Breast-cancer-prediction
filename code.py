import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE


breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

df = pd.DataFrame(breast_cancer_wisconsin_diagnostic.data.features)
df["target"] = breast_cancer_wisconsin_diagnostic.data.targets

label_encoder = LabelEncoder()
df["target"] = label_encoder.fit_transform(df["target"])
y = df['target']
X = df.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


corr_matrix = X_train.corr()
high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:  
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

X_train.drop(columns=high_corr_features, inplace=True)
X_test.drop(columns=high_corr_features, inplace=True)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

selector = RFE(estimator=xgb.XGBClassifier(random_state=42), n_features_to_select=20, step=1)
X_train_selected = selector.fit_transform(X_train_res, y_train_res)
X_test_selected = selector.transform(X_test)


my_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(random_state=42))
])

param_grid = {
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__gamma': [0, 0.1],
    'classifier__reg_alpha': [0, 0.1],
    'classifier__reg_lambda': [0, 0.1],
    'classifier__scale_pos_weight': [1, 2, 5]
}

grid_search = GridSearchCV(
    my_pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_selected, y_train_res)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_selected)

print(classification_report(y_test, y_pred))

y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")
