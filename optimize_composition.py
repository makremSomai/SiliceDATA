import pandas as pd
import joblib
import numpy as np
from scipy.optimize import minimize

# Load trained models and scaler
model_conductivity = joblib.load("model_conductivity.pkl")
model_resistance = joblib.load("model_resistance.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load feature names used in training

# Define constraints
bounds = [(5.00, 15.00), (5.00, 10.00), (35.00, 45.00), (20.00, 30.00), (15.00, 20.00)]
target_conductivity = 0.8
target_resistance = 60

# Initial guess
initial_guess = np.array([10.00, 7.00, 40.00, 25.00, 18.00], dtype=np.float64)

# Define objective function
def objective(params):
    # Ensure params is converted to DataFrame with correct feature names
    X = pd.DataFrame([params], columns=feature_names, dtype=np.float64)
    
    # Ensure column ordering matches training
    X = X[feature_names]

    # Apply correct scaling
    X_scaled = scaler.transform(X)
    conductivity = model_conductivity.predict(X_scaled)[0]
    resistance = model_resistance.predict(X_scaled)[0]
    
    return abs(conductivity - target_conductivity) + abs(resistance - target_resistance)

# Constraint: Sum must be 100
def constraint(params):
    return round(sum(params), 2) - 100.00

# Optimize
result = minimize(
    objective, 
    x0=initial_guess, 
    bounds=bounds, 
    constraints={"type": "eq", "fun": constraint}
)

# Convert optimized result to DataFrame
optimized_composition = pd.DataFrame([result.x], columns=feature_names)

print("Optimized Composition:\n", optimized_composition)
