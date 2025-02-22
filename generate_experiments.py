import pandas as pd
import joblib
import numpy as np
from scipy.stats.qmc import LatinHypercube

# Load trained models
model_conductivity = joblib.load("model_conductivity.pkl")
model_resistance = joblib.load("model_resistance.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # Load feature names

# Define constraints
bounds = {
    "Ciment (%)": (5.00, 15.00),
    "Argile (%)": (5.00, 10.00),
    "Alumine (%)": (35.00, 45.00),
    "Silice (%)": (20.00, 30.00),
    "Chamotte (%)": (15.00, 20.00),
}

num_samples = 10000  # Reduce sample size for faster performance

# Use Latin Hypercube Sampling (LHS) for diverse point generation
lhs = LatinHypercube(d=len(bounds))
samples = lhs.random(n=num_samples)

# Scale samples to match the given bounds
scaled_samples = np.array([
    np.round(bounds[key][0] + samples[:, i] * (bounds[key][1] - bounds[key][0]), 2)
    for i, key in enumerate(bounds.keys())
]).T  # Round to 2 decimal places

# Adjust compositions so that the sum is exactly 100%
scaled_samples = np.round((scaled_samples / np.sum(scaled_samples, axis=1)[:, None]) * 100, 2)

# Ensure constraints are still valid (values within min-max bounds)
for i, key in enumerate(bounds.keys()):
    scaled_samples[:, i] = np.clip(scaled_samples[:, i], bounds[key][0], bounds[key][1])

# Convert to DataFrame with correct feature names
df = pd.DataFrame(scaled_samples, columns=feature_names)

# Scale features for model input
X_scaled = scaler.transform(df)

# Predict Conductivity & Resistance (rounded to 2 decimal places)
df["Conductivite thermique (W/m.K)"] = np.round(model_conductivity.predict(X_scaled), 2)
df["Resistance Mecanique (MPa)"] = np.round(model_resistance.predict(X_scaled), 2)

# Sort by **lowest conductivity** while maintaining **high resistance**
df_sorted = df.sort_values(by=["Conductivite thermique (W/m.K)", "Resistance Mecanique (MPa)"], ascending=[True, False])

# Save new experiments
df_sorted.to_csv("generated_experiments.csv", index=False)

print(f"Generated {len(df_sorted)} optimized experiments with 2 decimal places and saved to generated_experiments.csv")
