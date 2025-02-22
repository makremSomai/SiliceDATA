import numpy as np
import pandas as pd
from scipy.stats.qmc import Halton
from sqlalchemy import create_engine

# Define constraints (min/max values)
bounds = {
    "Ciment": (5.00, 15.00),
    "Argile": (5.00, 10.00),
    "Alumine": (35.00, 45.00),
    "Silice": (20.00, 30.00),
    "Chamotte": (15.00, 20.00),
}

# Generate efficient samples using Halton Sampling
num_samples = 50_000  # Adjust this for performance vs. accuracy
sampler = Halton(d=len(bounds))  # d = number of components
samples = sampler.random(num_samples)

# Scale samples to match the given bounds
scaled_samples = []
for i, key in enumerate(bounds.keys()):
    min_val, max_val = bounds[key]
    scaled_samples.append(np.round(min_val + samples[:, i] * (max_val - min_val), 2))  # ✅ Limit to 2 decimals

scaled_samples = np.array(scaled_samples).T  # Convert list to NumPy array

# Keep only valid compositions where sum = 100
valid_compositions = scaled_samples[np.isclose(np.sum(scaled_samples, axis=1), 100, atol=0.01)]

# Convert to DataFrame
df = pd.DataFrame(valid_compositions, columns=bounds.keys())

# Save to CSV and database
df.to_csv("valid_compositions.csv", index=False)
engine = create_engine("sqlite:///experiments.db")
df.to_sql("experiments", engine, if_exists="replace", index=False)

print(f"Generated {len(df)} optimized compositions with 2 decimal places.")
