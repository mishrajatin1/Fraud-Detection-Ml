import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("Fraud.csv")
print(df[df['isFraud'] == 1].head(20))


x_features = [
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

y = 'isFraud'

# Label encode

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# save encoder
joblib.dump(le, "type_encoder.pkl")


# Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[x_features])

df_scaled = pd.DataFrame(scaled, columns=x_features)
joblib.dump(scaler,'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(df_scaled, df[y], test_size=0.3, random_state=42)

def metricse(test, predict):
    print("Precision Score :- ", precision_score(test, predict))
    print("Recall Score :- ", recall_score(test, predict))
    print("F1 Score :- ", f1_score(test, predict))





# ===============================
# XGBOOST
# ===============================
xgb = XGBClassifier(
    
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=5,
    eval_metric='auc',
    scale_pos_weight=250,
    tree_method = 'hist',
    device='cuda'    
)



xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=50
)
print("XGB Model Trained Successfully.")

xgb_probs = xgb.predict_proba(X_test)[:, 1]

# FINAL CHOSEN THRESHOLD (YOUR BEST)
THRESHOLD = 0.967
xgb_pred = (xgb_probs >= THRESHOLD).astype(int)

print("\n===== XGBOOST (Threshold =", THRESHOLD, ") =====")
metricse(y_test, xgb_pred)

cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title(f"Confusion Matrix - XGB (Threshold={THRESHOLD})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ===============================
# TIER LOGIC
# ===============================
def get_tier(prob):
    if prob < 0.05:
        return "approve"
    elif prob < 0.25:
        return "soft_flag"
    elif prob < THRESHOLD:
        return "manual_review"
    else:
        return "block"


# ===============================
# SAVE MODEL
# ===============================
joblib.dump(xgb, "fraud_model.pkl")
print("\nModel saved as fraud_model.pkl")
