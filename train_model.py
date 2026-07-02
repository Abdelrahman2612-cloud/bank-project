import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv("/mnt/user-data/uploads/bank.csv")

df['DepositFlag'] = np.where(df['deposit'] == 'yes', 1, 0)

cat_cols = ['job', 'marital', 'education', 'default', 'housing',
            'loan', 'contact', 'month', 'poutcome']
num_cols_lr = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

df_encoded = pd.get_dummies(df[cat_cols + num_cols_lr], drop_first=True)

X = df_encoded
y = df['DepositFlag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))

# Save model + the exact column structure needed for encoding new input
joblib.dump({
    "model": rf_model,
    "columns": X.columns.tolist(),
    "cat_cols": cat_cols,
    "num_cols": num_cols_lr,
    "cat_options": {c: sorted(df[c].unique().tolist()) for c in cat_cols}
}, "/home/claude/bank_model.pkl")

print("Saved model to bank_model.pkl")
