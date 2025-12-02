"""
Professional House Price Prediction Pipeline - Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
FEATURES = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd'
]
TARGET = 'SalePrice'


# ------------------------------
# Utility Functions
# ------------------------------
def clean_data(df, features, target=None):
    df = df.copy()
    for col in features:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    if target and target in df.columns:
        df[target].fillna(df[target].median(), inplace=True)
    return df


def evaluate_model(model, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


# ------------------------------
# Load Data
# ------------------------------
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train = clean_data(train_df, FEATURES, TARGET)
test = clean_data(test_df, FEATURES)

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]

# ------------------------------
# Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Train Model
# ------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
train_pred = model.predict(X_train_scaled)
rmse, r2 = evaluate_model(model, y_train, train_pred)
print(f"RMSE: {rmse:,.2f}  |  R²: {r2:.4f}")

# ------------------------------
# Submission File
# ------------------------------
test_pred = model.predict(X_test_scaled)
submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_pred})
submission.to_csv("submission.csv", index=False)

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(12, 8))

# 1 — Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_train, train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title("Actual vs Predicted")

# 2 — Residuals
residuals = y_train - train_pred
plt.subplot(2, 2, 2)
plt.scatter(train_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle='--', color='r')
plt.title("Residual Plot")

# 3 — Feature Importance
coef_df = pd.DataFrame({"Feature": FEATURES, "Importance": np.abs(model.coef_)})
coef_df = coef_df.sort_values("Importance", ascending=True)
plt.subplot(2, 2, 3)
plt.barh(coef_df["Feature"], coef_df["Importance"])
plt.title("Feature Importance")

# 4 — Distribution
plt.subplot(2, 2, 4)
plt.hist(y_train, bins=40, alpha=0.7, label="Actual")
plt.hist(train_pred, bins=40, alpha=0.7, label="Predicted")
plt.legend()
plt.title("Price Distribution")

plt.tight_layout()
plt.show()
