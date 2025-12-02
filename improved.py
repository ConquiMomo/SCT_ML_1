# IMPROVED House Price Prediction - Better Accuracy
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("=== IMPROVED House Price Prediction ===")

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Original data shapes: Train {train_df.shape}, Test {test_df.shape}")

# Select BETTER features based on correlation with price
# These are typically the most important for house prices
important_features = [
    'OverallQual',  # Overall quality (1-10) - MOST IMPORTANT!
    'GrLivArea',  # Living area
    'GarageCars',  # Garage size
    'GarageArea',  # Garage area
    'TotalBsmtSF',  # Total basement area
    '1stFlrSF',  # First floor area
    'FullBath',  # Full bathrooms
    'TotRmsAbvGrd',  # Total rooms
    'YearBuilt',  # Year built
    'YearRemodAdd'  # Remodel year
]

target = 'SalePrice'

# Only use features that exist
feature_columns = [f for f in important_features if f in train_df.columns]
print(f"Using {len(feature_columns)} important features: {feature_columns}")


# Data cleaning function
def clean_data(df, features, target=None):
    df_clean = df.copy()

    for col in features:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    if target and target in df_clean.columns:
        df_clean[target] = df_clean[target].fillna(df_clean[target].median())

    return df_clean


# Clean data
train_clean = clean_data(train_df, feature_columns, target)
test_clean = clean_data(test_df, feature_columns)

# Prepare data
X = train_clean[feature_columns]
y = train_clean[target]
X_test = test_clean[feature_columns]

print(f"\nData prepared: X {X.shape}, y {y.shape}")

# Split training data for validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n=== MODEL TRAINED ===")

# Make predictions
train_pred = model.predict(X_train_scaled)
val_pred = model.predict(X_val_scaled)

# Calculate metrics for both training and validation
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

train_r2 = r2_score(y_train, train_pred)
val_r2 = r2_score(y_val, val_pred)

median_price = y.median()

print("\n=== PERFORMANCE COMPARISON ===")
print(f"{'Metric':<15} {'Training':<12} {'Validation':<12} {'Improvement'}")
print(f"{'RMSE':<15} ${train_rmse:,.0f}     ${val_rmse:,.0f}     {((44356 - val_rmse) / 44356 * 100):+.1f}%")
print(f"{'R² Score':<15} {train_r2:.4f}      {val_r2:.4f}")
print(f"{'Error %':<15} {(train_rmse / median_price * 100):.1f}%        {(val_rmse / median_price * 100):.1f}%")

# Feature importance
print("\n=== FEATURE IMPORTANCE ===")
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_,
    'Impact': np.abs(model.coef_)
}).sort_values('Impact', ascending=False)

print(importance_df)

# Make final test predictions
test_predictions = model.predict(X_test_scaled)

# Create submission
submission_df = pd.DataFrame({
    'Id': test_clean['Id'],
    'SalePrice': test_predictions
})

submission_df.to_csv('improved_submission_v2.csv', index=False)
print(f"\nImproved predictions saved to 'improved_submission_v2.csv'")

# Show sample comparison
print("\n=== SAMPLE PREDICTIONS ===")
sample_data = pd.DataFrame({
    'Actual': y_val.head(10).values,
    'Predicted': val_pred[:10],
    'Error': y_val.head(10).values - val_pred[:10],
    'Error %': ((y_val.head(10).values - val_pred[:10]) / y_val.head(10).values * 100)
})
print(sample_data)

print(f"\n🎯 ORIGINAL MODEL RMSE: $44,356 (27% error)")
print(f"🎯 IMPROVED MODEL RMSE: ${val_rmse:,.0f} ({(val_rmse / median_price * 100):.1f}% error)")