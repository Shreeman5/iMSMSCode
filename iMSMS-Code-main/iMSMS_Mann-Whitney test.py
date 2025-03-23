import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
S1_PATH = 'iMSMS_dataset/Supplementary_Dataset_S1.xlsx'
S5_PATH = 'iMSMS_dataset/Supplementary_Dataset_S5.xlsx'
S6_PATH = 'iMSMS_dataset/Supplementary_Dataset_S6.xlsx'

# Load the necessary sheets
sheet1_1 = pd.read_excel(S1_PATH, sheet_name='Dataset S1.2')
sheet5_1 = pd.read_excel(S5_PATH, sheet_name='Dataset S5.1')
sheet6_phylum = pd.read_excel(S6_PATH, sheet_name='phylum')
sheet6_family = pd.read_excel(S6_PATH, sheet_name='family')
sheet6_class = pd.read_excel(S6_PATH, sheet_name='class')
sheet6_species = pd.read_excel(S6_PATH, sheet_name='species')

# Filter sheet1_1 to only include 'iMSMS_ID' and 'disease' columns
sheet1_1 = sheet1_1[['iMSMS_ID', 'disease']]
sheet5_1 = sheet5_1[['iMSMS_ID', 'chao1','shannon']]

# Merge the phylum abundance data with disease information
merged_df = pd.merge(sheet5_1, sheet1_1, on='iMSMS_ID')
#merged_df = pd.merge(sheet5_1, sheet1_1, on='iMSMS_ID')

# Get the list of phylum columns (excluding iMSMS_ID and disease)
phyla_columns = [col for col in merged_df.columns if col not in ['iMSMS_ID', 'disease']]

# Initialize lists to store results
p_values = []
bonferroni_p_values = []
fdr_p_values = []
phylum_names = []

# Perform Mann-Whitney U test for each phylum
for phylum in phyla_columns:
    ms_group = merged_df[merged_df['disease'] == 'MS'][phylum]
    control_group = merged_df[merged_df['disease'] == 'Control'][phylum]

    # Mann-Whitney U test
    stat, p_value = mannwhitneyu(ms_group, control_group, alternative='two-sided')

    p_values.append(p_value)
    phylum_names.append(phylum)

# Apply multiple comparison correction (FDR or Bonferroni)
# Bonferroni correction
bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]

# Benjamini-Hochberg (False Discovery Rate) correction
fdr_corrected = multipletests(p_values, method='fdr_bh')[1]

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Phylum': phylum_names,
    'P-value': p_values,
    'Bonferroni Corrected P-value': bonferroni_corrected,
    'FDR Corrected P-value': fdr_corrected
})

# Flag the significance for each p-value type (Raw, Bonferroni, FDR)
results_df['Raw Significant'] = results_df['P-value'].apply(lambda x: 1 if x <= 0.05 else 0)
results_df['Bonferroni Significant'] = results_df['Bonferroni Corrected P-value'].apply(lambda x: 1 if x <= 0.05 else 0)
results_df['FDR Significant'] = results_df['FDR Corrected P-value'].apply(lambda x: 1 if x <= 0.05 else 0)

# Prepare a DataFrame for the heatmap
# We'll combine the P-values, Bonferroni corrected, and FDR corrected values into one matrix.
heatmap_data = pd.DataFrame({
    'Class': phylum_names,
    'P-value': p_values,
    'Bonferroni P-value': bonferroni_corrected,
    'FDR P-value': fdr_corrected,
    'Raw Significant': results_df['Raw Significant'],
    'Bonferroni Significant': results_df['Bonferroni Significant'],
    'FDR Significant': results_df['FDR Significant']
})

# Set 'Phylum' as the index for heatmap visualization
heatmap_data.set_index('Class', inplace=True)

# Reshape the data into a matrix form suitable for a heatmap
# The columns of the matrix will be the three types of p-values (raw, Bonferroni, FDR)
heatmap_matrix = heatmap_data[['P-value', 'Bonferroni P-value', 'FDR P-value']].T  # Transpose to get phyla as rows and p-value types as columns
significance_matrix = heatmap_data[['Raw Significant', 'Bonferroni Significant', 'FDR Significant']].T  # Transpose significance flags

# Filter for p-values ≤ 0.05
heatmap_matrix_filtered = (heatmap_matrix.map(lambda x: x if x <= 0.05 else np.nan))
significance_matrix_filtered = significance_matrix.map(lambda x: x if x == 1 else np.nan)

print(heatmap_matrix_filtered)

# Plot the heatmap for filtered p-values
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_matrix_filtered, annot=True, cmap='coolwarm', cbar_kws={'label': 'P-value'}, fmt=".5f", linewidths=0.5,annot_kws={'rotation': 90})

# Title and layout
plt.title('P-values (≤0.05) for Mann-Whitney U Test Results (Raw, Bonferroni, FDR)')
plt.tight_layout()
plt.show()

# Plot the heatmap for filtered significance indicators
plt.figure(figsize=(10, 8))
sns.heatmap(significance_matrix_filtered, annot=True, cmap='Blues', cbar_kws={'label': 'Significance (1=Significant, 0=Not Significant)'}, fmt="d", linewidths=0.5)

# Title and layout for the significance heatmap
plt.title('Heatmap of Significance Indicators (≤0.05)')
plt.tight_layout()
plt.show()
