import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv("Task 3 and 4_Loan_data.csv")

X = df.drop(columns=["default", "customer_id"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pd = log_model.predict_proba(X_test_scaled)[:, 1]
log_auc = roc_auc_score(y_test, log_pd)


#random Forrest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pd = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pd)


#Neural Network
nn_model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
nn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["AUC"]
)
nn_model.fit(
    X_train_scaled, y_train,
    epochs=60,
    batch_size=16,
    verbose=0
)

nn_pd = nn_model.predict(X_test_scaled).flatten()
nn_auc = roc_auc_score(y_test, nn_pd)


#ensemble
avg_pd = (log_pd + rf_pd + nn_pd) / 3
avg_auc = roc_auc_score(y_test, avg_pd)

#stacking
stack = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    final_estimator=LogisticRegression()
)

stack.fit(X_train_scaled, y_train)
stack_pd = stack.predict_proba(X_test_scaled)[:, 1]
stack_auc = roc_auc_score(y_test, stack_pd)

def expected_loss(pd, loan_amt, recovery_rate=0.1):
    lgd = 1 - recovery_rate
    return pd * lgd * loan_amt


#EL
loan_amt = X_test["loan_amt_outstanding"].values
results = pd.DataFrame({
    "Logistic_EL": expected_loss(log_pd, loan_amt),
    "RandomForest_EL": expected_loss(rf_pd, loan_amt),
    "NeuralNet_EL": expected_loss(nn_pd, loan_amt),
    "Ensemble_EL": expected_loss(avg_pd, loan_amt),
    "Stacking_EL": expected_loss(stack_pd, loan_amt)
})

#performance comparison
print("AUC Scores:")
print(f"Logistic Regression: {log_auc:.4f}")
print(f"Random Forest:       {rf_auc:.4f}")
print(f"Neural Network:      {nn_auc:.4f}")
print(f"Ensemble (avg):      {avg_auc:.4f}")
print(f"Stacking model:      {stack_auc:.4f}")
print("\nExpected Loss Preview:")
print(results.head())

#AUC Scores:
#Logistic Regression: 0.99998839330128
#Random Forest:       0.99988310396286
#Neural Network:      0.99998839330128
#Ensemble (avg):      0.99998010280219
#Stacking model:      0.99998507710164
#
#Expected Loss Preview:
#    Logistic_EL  RandomForest_EL  NeuralNet_EL   Ensemble_EL  Stacking_EL
#0  9.617819e-10         0.000000  0.000000e+00  3.205940e-10     4.988816
#1  6.797764e-01        52.527213  3.365397e-08  1.773566e+01     5.963042
#2  1.293164e-05        14.765466  3.901614e-23  4.921826e+00     6.544856
#3  1.685374e-06         0.000000  1.172083e-26  5.617914e-07     4.795723
#4  1.565202e-06         0.000000  1.436055e-25  5.217341e-07     3.245998
#in conclusion, the ensemble and stacking models provide a more robust prediction of expected loss. random forest can have some extremes, logistic regression often produces very low expected loss as does the neural netwrok solution.