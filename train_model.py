import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("extracted_data.csv")

# Remove Experiment_ID if it exists
if "Experiment_ID" in df.columns:
    df = df.drop(columns=["Experiment_ID"])

# Define features (input) and targets (output)
X = df.iloc[:, :-2]  # All columns except last two (target variables)
y_conductivity = df.iloc[:, -2].round(2)  # Ensure only 2 decimal places
y_resistance = df.iloc[:, -1].round(2)  # Ensure only 2 decimal places

# Standardize feature names
feature_names = X.columns.tolist()

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train_c, y_test_c = train_test_split(X_scaled, y_conductivity, test_size=0.2, random_state=42)
X_train, X_test, y_train_r, y_test_r = train_test_split(X_scaled, y_resistance, test_size=0.2, random_state=42)

# Train models (Random Forest)
model_conductivity = RandomForestRegressor(n_estimators=200, random_state=42)
model_resistance = RandomForestRegressor(n_estimators=200, random_state=42)

model_conductivity.fit(X_train, y_train_c)
model_resistance.fit(X_train, y_train_r)

# Evaluate models
mse_c = mean_squared_error(y_test_c, model_conductivity.predict(X_test).round(2))
mse_r = mean_squared_error(y_test_r, model_resistance.predict(X_test).round(2))

print(f"Conductivity Model MSE: {mse_c:.4f}")
print(f"Resistance Model MSE: {mse_r:.4f}")

# Save models and scaler
joblib.dump(model_conductivity, "model_conductivity.pkl")
joblib.dump(model_resistance, "model_resistance.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")  # Save feature names
