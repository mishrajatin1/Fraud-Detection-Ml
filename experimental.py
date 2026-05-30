import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split  , GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score ,r2_score , confusion_matrix ,roc_auc_score,average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import precision_recall_curve



df=pd.read_csv("Fraud.csv")

# print("Data Structure:")
# print("\n",df.info())
# print("\n",df.describe())
# print("\n",df.isnull().sum())
# print(df.head())

x_features = [
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

y = 'isFraud'

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

scaler = StandardScaler()
scaled = scaler.fit_transform(df[x_features])

df_scaled = pd.DataFrame(scaled,columns=x_features)

# print(df_scaled.head(10))

def metricse(test,predict):
    
    print("Precision Score :- ",precision_score(test , predict))
    print("Recall Score :- ",recall_score(test , predict))
    print("F1  Score :- ",f1_score(test , predict))


# Scale properly
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df[y],   test_size=0.3,random_state=42)
               
                                                    

# Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', max_iter=500)
lr_model.fit(X_train, y_train)
lr_predict = lr_model.predict_proba(X_test)[:, 1]



precision, recall, thresholds = precision_recall_curve(y_test, lr_predict)
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

best_lr_threshold = thresholds[np.argmax(f1)]
best_lr_f1 = np.max(f1)

print("Best LR Threshold:", best_lr_threshold)
print("Best LR F1:", best_lr_f1) # best threshold 0.99


# Threshold tuning
threshold =0.5
lr_pred = (lr_predict >= threshold).astype(int)
print("\n--- Logistic Regression ---")
metricse(y_test, lr_pred)

cm= confusion_matrix(y_test , lr_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix-Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Random Forest (better params)
r_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

r_model.fit(X_train, y_train)
rf_probs = r_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, rf_probs)
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

best_rf_threshold = thresholds[np.argmax(f1)]
best_rf_f1 = np.max(f1) # got best threshold value 0.35

print("Best RF Threshold:", best_rf_threshold)
print("Best RF F1:", best_rf_f1)


#Threshold tuning
threshold = 0.5
rf_pred = (rf_probs >= threshold).astype(int)
print("\n--- Random Forest ---")
metricse(y_test, rf_pred)

cm= confusion_matrix(y_test , rf_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix-Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# ===============================
# 3. TRAIN XGBOOST MODEL
# ===============================

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    eval_metric='aucpr',
    scale_pos_weight=250
)

xgb.fit(X_train, y_train)
print("Model training completed.\n")

# ===============================
# 2. GET PREDICTION PROBABILITIES
# ===============================

probs = xgb.predict_proba(X_test)[:, 1]

# ===============================
# 3. APPLY FINAL THRESHOLD
# ===============================

THRESHOLD = 0.91
preds = (probs >= THRESHOLD).astype(int)

# ===============================
# 4. TIER SYSTEM FUNCTION
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

# Example:
# print(get_tier(0.82))

# ===============================
# 5. METRICS
# ===============================

prec = precision_score(y_test, preds)
rec  = recall_score(y_test, preds)
f1   = f1_score(y_test, preds)
aucroc = roc_auc_score(y_test, probs)
auprc = average_precision_score(y_test, probs)
cm = confusion_matrix(y_test, preds)

print("\n===== FINAL MODEL METRICS =====")
print("Threshold        :", THRESHOLD)
print("Precision        :", prec)
print("Recall           :", rec)
print("F1 Score         :", f1)
print("AUC-ROC          :", aucroc)
print("AUPRC (PR-AUC)   :", auprc)
print("\nConfusion Matrix:\n", cm)

# ===============================
# 6. CONFUSION MATRIX PLOT
# ===============================

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title(f"Confusion Matrix - XGB (Threshold={THRESHOLD})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 7. SAVE MODEL FOR DEPLOYMENT
# ===============================

joblib.dump(xgb, "fraud_model.pkl")
print("\nModel saved as fraud_model.pkl")

# ===============================
# 8. SAVE TIER LOGIC FOR DEPLOYMENT
# ===============================

# You’ll use this same function in FastAPI:
# prob = model.predict_proba(x)[0][1]
# action = get_tier(prob)
