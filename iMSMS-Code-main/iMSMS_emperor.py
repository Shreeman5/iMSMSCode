import pandas as pd
import numpy as np
from emperor import Emperor
from emperor.util import get_emperor_support_files_dir
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform
import os

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

# Define age colors
age_ranges = demographic_data['age'].unique().tolist()
age_ranges = [x for x in age_ranges if not pd.isna(x)]
age_ranges.sort()  # Sort the ranges

# Define a custom color map for the age ranges
custom_colors = {
    age_ranges[0]: '#1f77b4',  # blue
    age_ranges[1]: '#2ca02c',  # green
    age_ranges[2]: '#ffff00',  # yellow
    age_ranges[3]: '#ff7f0e',  # orange
    age_ranges[4]: '#d62728'   # red
}

# Create the Emperor visualization
viz = Emperor(pcoa_results, demographic_data, remote=get_emperor_support_files_dir())

# Use Emperor's color_by method to set the initial coloring
viz.color_by('age', custom_colors)

# Set other visualization options
viz.set_axes([0, 1, 2])  # Set axes to display (using indices 0, 1, 2 for pc1, pc2, pc3)

# Create dictionaries for scaling and opacity
scale_dict = {age_range: 1.0 for age_range in age_ranges}
opacity_dict = {age_range: 1.0 for age_range in age_ranges}

# Set scaling and opacity
viz.scale_by('age', scale_dict)
viz.opacity_by('age', opacity_dict)

# Generate the base Emperor visualization HTML
emperor_html = viz.make_emperor(standalone=True)

# Define the exact JavaScript code that should be used to replace the custom code marker
custom_js = """
// AUTO-SELECT AGE FOR COLORING
// This will select 'age' as the coloring variable when the visualization loads
// Find the color controller and select 'age'
if (ec.controllers && ec.controllers.color) {
  // Set the metadata field to 'age'
  setTimeout(function() {
    // Set 'age' as the coloring category if available
    var controller = ec.controllers.color;
    
    // First check if 'age' is available in the dropdown
    var select = controller.$select[0];
    var hasAge = false;
    
    for (var i = 0; i < select.options.length; i++) {
      if (select.options[i].value === 'age') {
        hasAge = true;
        select.selectedIndex = i;
        
        // Trigger change event to apply the selection
        $(select).trigger('change');
        console.log('Auto-selected age for coloring');
        break;
      }
    }
    
    if (!hasAge) {
      console.log('Age category not found in available metadata');
    }
    
    // Custom colors for age ranges - try to apply them if possible
    // These map to the 5 age ranges in your data
    var customColors = {
      0: '#1f77b4', // blue
      1: '#2ca02c', // green
      2: '#ffff00', // yellow
      3: '#ff7f0e', // orange
      4: '#d62728' // red
    };
    
    // Attempt to set custom colors if the editor exists
    if (controller.colorEditor) {
      // Try to set colors for each value
      for (var value in customColors) {
        controller.colorEditor.setValueColor(value, customColors[value]);
      }
    }
  }, 1000); // Small delay to ensure controllers are fully initialized
}
"""

# Look for the specific marker pattern in the HTML
marker_pattern = "/*__custom_on_ready_code__*/"
if marker_pattern in emperor_html:
    # Replace the marker with our custom JavaScript
    modified_html = emperor_html.replace(marker_pattern, marker_pattern + "\n      " + custom_js)
else:
    # If the marker isn't found, find a suitable insertion point
    # Look for the end of the ec.ready function
    ready_function_end = "ec.ready = function () {"
    ready_end_pattern = "}"
    
    # Find the ec.ready function
    start_idx = emperor_html.find(ready_function_end)
    if start_idx != -1:
        # Find the corresponding closing bracket
        bracket_count = 1
        idx = start_idx + len(ready_function_end)
        while idx < len(emperor_html) and bracket_count > 0:
            if emperor_html[idx] == '{':
                bracket_count += 1
            elif emperor_html[idx] == '}':
                bracket_count -= 1
            idx += 1
        
        # Insert our code just before the closing bracket
        if bracket_count == 0 and idx > 0:
            insertion_idx = idx - 1
            modified_html = emperor_html[:insertion_idx] + "\n      " + custom_js + "\n    " + emperor_html[insertion_idx:]
        else:
            # If we can't find the right spot, just append the code to the ec.ready function
            modified_html = emperor_html.replace("ec.ready = function () {", 
                                            "ec.ready = function () {\n      " + custom_js)
    else:
        # If we can't find ec.ready, insert the code at a good fallback location
        modified_html = emperor_html
        print("Warning: Could not find a suitable location to insert custom JavaScript.")

# Convert absolute paths to relative paths
support_dir = get_emperor_support_files_dir()
if support_dir in modified_html:
    relative_html = modified_html.replace(support_dir + '/', '')
else:
    relative_html = modified_html

# Ensure the output directory exists
import os
output_dir = "iMSMS_emperor_host_static_html"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write the relative path HTML directly to emperor.html in the specified folder
output_path = os.path.join(output_dir, "emperor.html")
with open(output_path, 'w') as f:
    f.write(relative_html)

print(f"Emperor visualization saved to {output_path} with relative paths and age coloring.")

print("Emperor visualization saved with age coloring.")

print("Script completed.")
