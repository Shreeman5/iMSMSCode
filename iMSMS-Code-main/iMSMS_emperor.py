import pandas as pd
import numpy as np
from emperor import Emperor
from emperor.util import get_emperor_support_files_dir
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform

# Function that converts integer columns to range values
def convert_to_ranges(df, num_bins=5):
    def categorize_value(value, min_val, max_val, range_steps, range_labels):
        if pd.isna(value):
            return np.nan
        for i in range(len(range_steps) - 1):
            if range_steps[i] <= value < range_steps[i + 1]:
                return range_labels[i]
        return range_labels[-1]
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        range_steps = np.linspace(min_val, max_val, num_bins + 1)
        range_labels = [f"{round(range_steps[i], 2)}-{round(range_steps[i + 1], 2)}" for i in range(len(range_steps) - 1)]
        df[col] = df[col].apply(categorize_value, args=(min_val, max_val, range_steps, range_labels))
    return df

# Load your data
S1_PATH = 'iMSMS_dataset/Supplementary_Dataset_S1.xlsx'
S2_PATH = 'iMSMS_dataset/Supplementary_Dataset_S2.xlsx'
S3_PATH = 'iMSMS_dataset/Supplementary_Dataset_S3.xlsx'
S6_PATH = 'iMSMS_dataset/Supplementary_Dataset_S6.xlsx'

sheet1_2 = pd.read_excel(S1_PATH, sheet_name='Dataset S1.2')
sheet2 = pd.read_excel(S2_PATH, sheet_name='Dataset S2')
sheet3 = pd.read_excel(S3_PATH, sheet_name='Dataset S3')
sheet6_class = pd.read_excel(S6_PATH, sheet_name='class')

# Merge the demographic data
demographic_data = (sheet1_2
                   .merge(sheet3, on='iMSMS_ID', how='inner')
                   .merge(sheet2, on='iMSMS_ID', how='inner')
                   )

# Convert numeric columns to ranges
demographic_data = convert_to_ranges(demographic_data, num_bins=5)
sheet6_class = sheet6_class.merge(demographic_data[['iMSMS_ID']], on='iMSMS_ID', how='inner')
demographic_data = demographic_data.set_index('iMSMS_ID')
sheet6_class = sheet6_class.set_index('iMSMS_ID')

# Beta Diversity
bray_curtis = pdist(sheet6_class, metric='braycurtis')
bray_curtis_matrix = squareform(bray_curtis)
bray_curtis_df = pd.DataFrame(bray_curtis_matrix, index=sheet6_class.index, columns=sheet6_class.index)

# Prepare distance matrix
distance_matrix = bray_curtis_df.to_numpy()
distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix, 0)

# Perform PCoA
pcoa_results = pcoa(distance_matrix)

# Fix sample IDs if needed
if isinstance(pcoa_results.samples, pd.DataFrame):
    pcoa_results.samples.index = demographic_data.index
else:
    pcoa_results.samples = pd.DataFrame(
        data=pcoa_results.samples,
        index=demographic_data.index
    )

# Create the Emperor visualization
viz = Emperor(pcoa_results, demographic_data, remote=get_emperor_support_files_dir())

# Define age colors
age_ranges = demographic_data['age'].unique().tolist()
age_ranges = [x for x in age_ranges if not pd.isna(x)]
age_ranges.sort()  # Sort the ranges

color_map = {
    age_ranges[0]: '#1f77b4',  # blue
    age_ranges[1]: '#2ca02c',  # green
    age_ranges[2]: '#ffff00',  # yellow
    age_ranges[3]: '#ff7f0e',  # orange
    age_ranges[4]: '#d62728'   # red
}

# Set the visualization options using the available methods
# Use the color_by method to set coloring to age
viz.color_by('age', color_map)

# Set other visualization options
viz.set_axes([0, 1, 2])  # Set axes to display (using indices 0, 1, 2 for pc1, pc2, pc3)

# Create dictionaries for scaling and opacity (mapping each age range to a value)
scale_dict = {age_range: 1.0 for age_range in age_ranges if not pd.isna(age_range)}
opacity_dict = {age_range: 1.0 for age_range in age_ranges if not pd.isna(age_range)}

# Set scaling and opacity
viz.scale_by('age', scale_dict)
viz.opacity_by('age', opacity_dict)

# Save the visualization to an HTML file
with open('iMSMS-emperor-configured.html', 'w') as f:
    f.write(viz.make_emperor(standalone=True))

print("Emperor visualization saved with preset colors for age variable.")
