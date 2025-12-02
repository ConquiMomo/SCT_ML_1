# Task 01: House Price Prediction with Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=== House Price Prediction Model (Task 01) ===")

# Step 1: Load the data
print("\n1. Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

# Step 2: Select features (square footage, bedrooms, bathrooms)
print("\n2. Selecting features...")
selected_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']  # adjust names if dataset differs
target_column = 'SalePrice'

print(f"Using features: {selected_features}")

# Step 3: Clean data (fill missing values with median)
print("\n3. Cleaning data...")
def clean_data(df, feature_columns, target_column=None):
    df_clean = df.copy()
    for col in feature_columns:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"  Filled {col} with median: {median_val}")
    if target_column and target_column in df_clean.columns:
        df_clean[target_column] = df_clean[target_column].fillna(df_clean[target_column].median())
    return df_clean

train_clean = clean_data(train_df, selected_features, target_column)
test_clean = clean_data(test_df, selected_features)

# Step 4: Prepare data
print("\n4. Preparing data...")
X_train = train_clean[selected_features]
y_train = train_clean[target_column]
X_test = test_clean[selected_features]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Step 5: Scale features
print("\n5. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed")

# Step 6: Train model
print("\n6. Training model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Model trained successfully!")

# Step 7: Evaluate model
print("\n7. Evaluating model...")
train_predictions = model.predict(X_train_scaled)

mse = mean_squared_error(y_train, train_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, train_predictions)

print(f"Training RMSE: ${rmse:,.2f}")
print(f"Training R² Score: {r2:.4f}")

# Step 8: Make test predictions
print("\n8. Making test predictions...")
test_predictions = model.predict(X_test_scaled)

# Step 9: Create submission file
print("\n9. Creating submission file...")
submission_df = pd.DataFrame({
    'Id': test_clean['Id'],
    'SalePrice': test_predictions
})
submission_df.to_csv('submission_task01.csv', index=False)
print("Submission file saved as 'submission_task01.csv'")

# Step 10: Visualization
print("\n10. Visualization...")
plt.figure(figsize=(12, 6))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_train, train_predictions, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')

# Residuals
plt.subplot(1, 2, 2)
residuals = y_train - train_predictions
plt.scatter(train_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.savefig('analysis_task01.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 Task 01 model complete! Check 'submission_task01.csv' and 'analysis_task01.png'")