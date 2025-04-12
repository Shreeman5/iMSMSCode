import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_people = 100
data = {
    "MS_status": np.random.choice(["yes", "no"], size=num_people),
    "smoking_status": np.random.choice(["current", "former", "no smoking"], size=num_people),
    "salt_consumption": np.random.uniform(0, 100, size=num_people),  # in mg per day
    "EDSS_score": np.random.uniform(0, 10, size=num_people)  # EDSS score range
}

# Create DataFrame
df = pd.DataFrame(data)

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cats = encoder.fit_transform(df[['MS_status', 'smoking_status']])

# Standardize numerical variables
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[['salt_consumption', 'EDSS_score']])

# Combine encoded categorical and scaled numerical data
processed_data = np.hstack([encoded_cats, scaled_nums])

# Compute Gower's distance matrix using Euclidean distance (approximation)
gower_dist = squareform(pdist(processed_data, metric='euclidean'))

# Perform PCoA on the Gower's distance matrix
ordination_results = pcoa(gower_dist)

# Convert the ordination coordinates to a DataFrame for visualization
ordination_df = pd.DataFrame(ordination_results.samples.iloc[:, :3], columns=["PC1", "PC2", "PC3"])

# Add metadata back to the ordination results
ordination_df["MS_status"] = df["MS_status"]
ordination_df["smoking_status"] = df["smoking_status"]
ordination_df["EDSS_score"] = df["EDSS_score"]

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define file paths for saving the CSV files
ordination_coords_path = os.path.join(current_directory, 'ordination_coords.csv')
metadata_path = os.path.join(current_directory, 'metadata.csv')

# Save ordination coordinates to CSV
ordination_df[['PC1', 'PC2', 'PC3']].to_csv(ordination_coords_path, index=False)

# Save metadata to CSV (including the relevant columns for coloring)
ordination_df[['smoking_status', 'EDSS_score']].to_csv(metadata_path, index=False)

# Output file paths for user reference
print(f"CSV files saved to {current_directory}:")
print(f"1. {ordination_coords_path} (PCoA coordinates)")
print(f"2. {metadata_path} (smoking status and EDSS score)")