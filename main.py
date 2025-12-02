# Improved House Price Prediction Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=== IMPROVED House Price Prediction Model ===")

# Step 1: Load the data
print("\n1. Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

# Step 2: Select BETTER features based on common real estate factors
print("\n2. Selecting improved features...")


improved_features = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'GarageArea',
    '1stFlrSF',
    'FullBath',
    'TotRmsAbvGrd',
    'YearBuilt',
    'YearRemodAdd'
]

target_column = 'SalePrice'

# Only use features that exist in our data
feature_columns = [col for col in improved_features if col in train_df.columns]
print(f"Using {len(feature_columns)} features: {feature_columns}")

# Step 3: Data cleaning with better handling
print("\n3. Cleaning data...")


def clean_data_improved(df, feature_columns, target_column=None):
    df_clean = df.copy()

    # Fill missing values with median for numerical columns
    for col in feature_columns:
        if col in df_clean.columns:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"  Filled {col} with median: {median_val}")

    # Fill target if provided
    if target_column and target_column in df_clean.columns:
        df_clean[target_column] = df_clean[target_column].fillna(df_clean[target_column].median())

    return df_clean


# Clean the data
train_clean = clean_data_improved(train_df, feature_columns, target_column)
test_clean = clean_data_improved(test_df, feature_columns)

# Step 4: Prepare data
print("\n4. Preparing data...")
X_train = train_clean[feature_columns]
y_train = train_clean[target_column]
X_test = test_clean[feature_columns]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Step 5: Feature scaling (improves model performance)
print("\n5. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

# Step 6: Create and train improved model
print("\n6. Training improved model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

# Step 7: Make predictions and evaluate
print("\n7. Evaluating model...")
train_predictions = model.predict(X_train_scaled)

# Calculate multiple metrics
mse = mean_squared_error(y_train, train_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, train_predictions)

# Calculate percentage error
median_price = y_train.median()
percentage_error = (rmse / median_price) * 100

print(f"Training RMSE: ${rmse:,.2f}")
print(f"Training R² Score: {r2:.4f}")
print(f"Median House Price: ${median_price:,.2f}")
print(f"Average Error: {percentage_error:.1f}% of median price")

# Step 8: Make test predictions
print("\n8. Making test predictions...")
test_predictions = model.predict(X_test_scaled)

# Step 9: Create improved submission
print("\n9. Creating submission file...")
submission_df = pd.DataFrame({
    'Id': test_clean['Id'],
    'SalePrice': test_predictions
})

submission_df.to_csv('improved_submission.csv', index=False)
print("Improved submission file saved as 'improved_submission.csv'")

# Step 10: Enhanced analysis
print("\n10. Enhanced Analysis")
print("\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_,
    'Abs_Effect': np.abs(model.coef_)
}).sort_values('Abs_Effect', ascending=False)

print(feature_importance)

# Visualization
plt.figure(figsize=(12, 8))

# Plot 1: Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(y_train, train_predictions, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')

# Plot 2: Residuals
plt.subplot(2, 2, 2)
residuals = y_train - train_predictions
plt.scatter(train_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 3: Feature importance
plt.subplot(2, 2, 3)
plt.barh(feature_importance['Feature'], feature_importance['Abs_Effect'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance')

# Plot 4: Price distribution
plt.subplot(2, 2, 4)
plt.hist(y_train, bins=50, alpha=0.7, label='Actual')
plt.hist(train_predictions, bins=50, alpha=0.7, label='Predicted')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.title('Price Distribution')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Sample Predictions ===")
sample_comparison = pd.DataFrame({
    'Actual': y_train.head(10),
    'Predicted': train_predictions[:10],
    'Error': y_train.head(10) - train_predictions[:10]
})
print(sample_comparison)

print(f"\n=== Summary ===")
print(f"Original Model RMSE: $44,356")
print(f"Improved Model RMSE: ${rmse:,.2f}")
improvement = ((44356 - rmse) / 44356) * 100
print(f"Improvement: {improvement:+.1f}%")

print("\n🎉 Improved model complete! Check 'improved_submission.csv' and 'model_analysis.png'")