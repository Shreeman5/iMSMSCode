import pandas as pd
import numpy as np
from emperor import Emperor
from emperor.util import get_emperor_support_files_dir
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import pdist, squareform
import os
import re

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

# Function to convert height specifically to 4 bins
def convert_height_to_ranges(df, num_bins=4):
    if 'height' not in df.columns:
        print("Warning: 'height' column not found in data.")
        return df
    
    # Check if height is already converted to categories
    if df['height'].dtype == 'object' or df['height'].dtype.name.startswith('str'):
        print("Warning: 'height' column is not numeric. Creating artificial bins.")
        
        # Get unique values and create artificial bins
        unique_heights = df['height'].dropna().unique()
        
        if len(unique_heights) <= num_bins:
            # If we have fewer unique values than bins, use the values directly
            range_labels = [f"height_bin{i+1}" for i in range(len(unique_heights))]
            height_map = {val: range_labels[i] for i, val in enumerate(sorted(unique_heights))}
            
            df['height_bins'] = df['height'].map(height_map)
        else:
            # Create artificial bins by splitting unique values into num_bins groups
            import numpy as np
            
            # Sort unique values
            sorted_heights = sorted(unique_heights)
            
            # Split into equal sized groups
            splits = np.array_split(sorted_heights, num_bins)
            
            # Create mapping from height values to bins
            height_map = {}
            for i, split in enumerate(splits):
                for val in split:
                    height_map[val] = f"height_bin{i+1}"
            
            # Apply mapping
            df['height_bins'] = df['height'].map(height_map)
    else:
        # Original numeric processing
        min_val = df['height'].min()
        max_val = df['height'].max()
        range_steps = np.linspace(min_val, max_val, num_bins + 1)
        range_labels = [f"height_bin{i+1}" for i in range(num_bins)]
        
        # Create a new column for height bins
        df['height_bins'] = np.nan
        
        for i in range(len(range_steps) - 1):
            mask = (df['height'] >= range_steps[i]) & (df['height'] < range_steps[i + 1])
            df.loc[mask, 'height_bins'] = range_labels[i]
        
        # Handle edge case for maximum value
        df.loc[df['height'] == max_val, 'height_bins'] = range_labels[-1]
    
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

# Print height column info for debugging
if 'height' in demographic_data.columns:
    print(f"Height column data type: {demographic_data['height'].dtype}")
    print(f"Height column unique values (sample): {demographic_data['height'].sample(min(5, len(demographic_data))).tolist()}")
    print(f"Number of unique height values: {demographic_data['height'].nunique()}")
else:
    print("Warning: 'height' column not found. Available columns:", demographic_data.columns.tolist())
    # If height column is not available, create a mock height column for demonstration
    print("Creating a mock 'height' column for demonstration purposes.")
    import numpy as np
    demographic_data['height'] = np.random.randint(150, 190, size=len(demographic_data))

# Convert age to 5 bins
demographic_data = convert_to_ranges(demographic_data, num_bins=5)

# Convert height to 4 bins
demographic_data = convert_height_to_ranges(demographic_data, num_bins=4)

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

# Define height shapes
height_bins = ["height_bin1", "height_bin2", "height_bin3", "height_bin4"]
custom_shapes = {
    "height_bin1": "Cone",      # 1st bin: Cone
    "height_bin2": "Star",      # 2nd bin: Star
    "height_bin3": "Cylinder",  # 3rd bin: Cylinder
    "height_bin4": "Sphere"     # 4th bin: Sphere
}

# Add a more aggressive approach to rename the axis labels
# First, attempt to rename in the decomposition data itself
if hasattr(pcoa_results, 'samples') and isinstance(pcoa_results.samples, pd.DataFrame):
    # Rename the column labels if they exist
    if pcoa_results.samples.columns.tolist():
        new_columns = []
        for col in pcoa_results.samples.columns:
            if 'PC1' in str(col) or 'pc1' in str(col).lower():
                new_columns.append('Axis 1')
            elif 'PC2' in str(col) or 'pc2' in str(col).lower():
                new_columns.append('Axis 2')
            elif 'PC3' in str(col) or 'pc3' in str(col).lower():
                new_columns.append('Axis 3')
            else:
                new_columns.append(col)
        pcoa_results.samples.columns = new_columns

# Create the Emperor visualization
viz = Emperor(pcoa_results, demographic_data, remote=get_emperor_support_files_dir())

# Use Emperor's color_by method to set the initial coloring
viz.color_by('age', custom_colors)

# Use Emperor's shape_by method to set the shapes by height
viz.shape_by('height_bins', custom_shapes)

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

# Create the direct shape override JavaScript
direct_shape_js = """// This script directly manipulates the THREE.js objects in Emperor
// Map of shape names to THREE.js geometry creation functions
const SHAPE_GEOMETRIES = {
  'Cone': function() {
    return new THREE.ConeGeometry(0.5, 1, 8);
  },
  'Sphere': function() {
    return new THREE.SphereGeometry(0.5, 16, 16);
  },
  'Star': function() {
    // Create a star shape
    const starShape = new THREE.Shape();
    const outerRadius = 0.5;
    const innerRadius = 0.2;
    const spikes = 5;
    
    for (let i = 0; i < spikes * 2; i++) {
      const radius = i % 2 === 0 ? outerRadius : innerRadius;
      const angle = (Math.PI * 2 * i) / (spikes * 2);
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      
      if (i === 0) {
        starShape.moveTo(x, y);
      } else {
        starShape.lineTo(x, y);
      }
    }
    starShape.closePath();
    
    const extrudeSettings = {
      depth: 0.2,
      bevelEnabled: false
    };
    
    return new THREE.ExtrudeGeometry(starShape, extrudeSettings);
  },
  'Cylinder': function() {
    return new THREE.CylinderGeometry(0.4, 0.4, 0.8, 16);
  }
};

// Function to directly replace geometries in THREE.js scene
function replaceGeometries() {
  // Accessing Emperor's controller
  if (typeof empObj === 'undefined' || !empObj.sceneViews || !empObj.sceneViews[0]) {
    console.log("Emperor or scene view not available yet");
    return false;
  }
  
  const sceneView = empObj.sceneViews[0];
  const scene = sceneView.scene;
  const metadata = window.data.plot.metadata;
  const metadataHeaders = window.data.plot.metadata_headers;
  
  // Find height_bins index in metadata
  let heightBinsIndex = -1;
  for (let i = 0; i < metadataHeaders.length; i++) {
    if (metadataHeaders[i] === 'height_bins') {
      heightBinsIndex = i;
      break;
    }
  }
  
  if (heightBinsIndex === -1) {
    console.log("height_bins field not found in metadata");
    return false;
  }
  
  // Create a map from sample ID to height_bins value
  const sampleToHeightBin = {};
  for (const sampleId in metadata) {
    if (metadata.hasOwnProperty(sampleId)) {
      sampleToHeightBin[sampleId] = metadata[sampleId][heightBinsIndex];
    }
  }
  
  // Map height_bins values to shape types
  const heightBinToShape = {
    'height_bin1': 'Cone',
    'height_bin2': 'Star',
    'height_bin3': 'Cylinder',
    'height_bin4': 'Sphere'
  };
  
  // Go through all objects in the scene
  let pointsReplaced = false;
  
  try {
    // First, try to replace THREE.Points with individual meshes
    for (let i = 0; i < scene.children.length; i++) {
      const child = scene.children[i];
      
      if (child instanceof THREE.Points) {
        console.log("Found points object:", child);
        
        // Get the positions from the points
        const positions = child.geometry.attributes.position.array;
        const colors = child.geometry.attributes.color.array;
        const count = child.geometry.attributes.position.count;
        
        // Create a parent object to hold our meshes
        const meshesParent = new THREE.Object3D();
        meshesParent.name = "CustomShapesContainer";
        
        // Get sample IDs if available
        let sampleIds = [];
        if (sceneView.decomp && sceneView.decomp.plottable && sceneView.decomp.plottable.sample_ids) {
          sampleIds = sceneView.decomp.plottable.sample_ids;
        }
        
        // Replace points with appropriate geometries
        for (let j = 0; j < count; j++) {
          const x = positions[j * 3];
          const y = positions[j * 3 + 1];
          const z = positions[j * 3 + 2];
          
          const r = colors[j * 3];
          const g = colors[j * 3 + 1];
          const b = colors[j * 3 + 2];
          
          // Determine which shape to use based on sample ID and height_bins
          let shapeName = 'Sphere'; // Default
          
          if (j < sampleIds.length) {
            const sampleId = sampleIds[j];
            const heightBin = sampleToHeightBin[sampleId];
            
            if (heightBin && heightBinToShape[heightBin]) {
              shapeName = heightBinToShape[heightBin];
            }
          }
          
          // Create geometry based on shape type
          const geometryFunc = SHAPE_GEOMETRIES[shapeName];
          if (!geometryFunc) continue;
          
          const geometry = geometryFunc();
          const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(r, g, b),
            transparent: true,
            opacity: 0.8
          });
          
          const mesh = new THREE.Mesh(geometry, material);
          mesh.position.set(x, y, z);
          mesh.scale.set(0.3, 0.3, 0.3); // Adjust size as needed
          
          meshesParent.add(mesh);
        }
        
        // Add the new meshes and remove (or hide) the original points
        scene.add(meshesParent);
        child.visible = false; // Hide instead of remove to preserve original data
        
        pointsReplaced = true;
        console.log(`Replaced ${count} points with custom geometries`);
      }
    }
    
    // Force a redraw
    if (sceneView.needsUpdate !== undefined) {
      sceneView.needsUpdate = true;
    }
    
    // Also try to trigger Emperor's render function
    if (empObj.drawFrame) {
      empObj.drawFrame();
    }
    
    return pointsReplaced;
  } catch (e) {
    console.error("Error replacing geometries:", e);
    return false;
  }
}

// Initialize after page load
document.addEventListener('DOMContentLoaded', function() {
  console.log("Direct THREE.js shape override script loaded");
  
  // Wait for Emperor to fully initialize
  let attempts = 0;
  const maxAttempts = 20;
  
  function tryReplaceGeometries() {
    attempts++;
    console.log(`Attempt ${attempts} to replace geometries`);
    
    const success = replaceGeometries();
    
    if (success) {
      console.log("Successfully replaced geometries!");
    } else if (attempts < maxAttempts) {
      // Try again after a delay
      setTimeout(tryReplaceGeometries, 1000);
    } else {
      console.log(`Failed to replace geometries after ${maxAttempts} attempts`);
    }
  }
  
  // Start first attempt after 3 seconds to ensure Emperor is fully loaded
  setTimeout(tryReplaceGeometries, 3000);
});
"""

# Also add standard JavaScript to select age for coloring
custom_js = """
// Set the metadata field to 'age' for coloring
setTimeout(function() {
  // HANDLE COLORING BY AGE
  if (ec.controllers && ec.controllers.color) {
    var colorController = ec.controllers.color;
    
    // First check if 'age' is available in the dropdown
    var colorSelect = colorController.$select[0];
    var hasAge = false;
    
    for (var i = 0; i < colorSelect.options.length; i++) {
      if (colorSelect.options[i].value === 'age') {
        hasAge = true;
        colorSelect.selectedIndex = i;
        
        // Trigger change event to apply the selection
        $(colorSelect).trigger('change');
        console.log('Auto-selected age for coloring');
        break;
      }
    }
    
    if (!hasAge) {
      console.log('Age category not found in available metadata');
    }
    
    // Custom colors for age ranges
    var customColors = {
      0: '#1f77b4', // blue
      1: '#2ca02c', // green
      2: '#ffff00', // yellow
      3: '#ff7f0e', // orange
      4: '#d62728' // red
    };
    
    // Attempt to set custom colors if the editor exists
    if (colorController.colorEditor) {
      // Try to set colors for each value
      for (var value in customColors) {
        colorController.colorEditor.setValueColor(value, customColors[value]);
      }
    }
  }
  
  // RENAME AXIS LABELS
  function renameAxisLabels() {
    // Method 1: Look for text elements in the SVG renderer
    var allText = document.querySelectorAll('text');
    for (var i = 0; i < allText.length; i++) {
      var text = allText[i].textContent || allText[i].innerText;
      if (text.match(/PC1/i) || text.match(/PC 1/i)) {
        allText[i].textContent = text.replace(/PC1/i, 'Axis 1').replace(/PC 1/i, 'Axis 1');
      }
      if (text.match(/PC2/i) || text.match(/PC 2/i)) {
        allText[i].textContent = text.replace(/PC2/i, 'Axis 2').replace(/PC 2/i, 'Axis 2');
      }
      if (text.match(/PC3/i) || text.match(/PC 3/i)) {
        allText[i].textContent = text.replace(/PC3/i, 'Axis 3').replace(/PC 3/i, 'Axis 3');
      }
    }
  }
  
  // Run the axis renaming function repeatedly
  renameAxisLabels();
  
  var renameInterval = setInterval(renameAxisLabels, 500);
  
  // Stop trying after 10 seconds
  setTimeout(function() {
    clearInterval(renameInterval);
  }, 10000);
}, 1000);
"""

# Look for the specific marker pattern in the HTML
marker_pattern = "/*__custom_on_ready_code__*/"
if marker_pattern in emperor_html:
    # Replace the marker with our custom JavaScript
    emperor_html = emperor_html.replace(marker_pattern, marker_pattern + "\n      " + custom_js)
else:
    # If the marker isn't found, find a suitable insertion point
    ready_function_end = "ec.ready = function () {"
    
    # Find the ec.ready function
    start_idx = emperor_html.find(ready_function_end)
    if start_idx != -1:
        # Insert our code after the opening brace of ec.ready
        insertion_idx = start_idx + len(ready_function_end)
        emperor_html = emperor_html[:insertion_idx] + "\n      " + custom_js + emperor_html[insertion_idx:]
    else:
        print("Warning: Could not find a suitable location to insert custom JavaScript.")

# Also perform more aggressive text replacement in the HTML
# This searches for any instances of PC1, PC2, PC3 with different capitalizations and spacings
def replace_pc_labels(html):
    # Define patterns and replacements with regex
    replacements = [
        (r'PC1(?!\d)', 'Axis 1'),
        (r'PC 1(?!\d)', 'Axis 1'),
        (r'pc1(?!\d)', 'Axis 1'),
        (r'Pc1(?!\d)', 'Axis 1'),
        (r'PC2(?!\d)', 'Axis 2'),
        (r'PC 2(?!\d)', 'Axis 2'),
        (r'pc2(?!\d)', 'Axis 2'),
        (r'Pc2(?!\d)', 'Axis 2'),
        (r'PC3(?!\d)', 'Axis 3'),
        (r'PC 3(?!\d)', 'Axis 3'),
        (r'pc3(?!\d)', 'Axis 3'),
        (r'Pc3(?!\d)', 'Axis 3'),
    ]
    
    import re
    modified = html
    for pattern, replacement in replacements:
        modified = re.sub(pattern, replacement, modified)
    
    return modified

# Apply the regex replacements
emperor_html = replace_pc_labels(emperor_html)

# Insert the direct shape manipulation script at the end of the HTML body
emperor_html = emperor_html.replace('</body>', f'<script type="text/javascript">{direct_shape_js}</script></body>')

# Convert absolute paths to relative paths
support_dir = get_emperor_support_files_dir()
if support_dir in emperor_html:
    emperor_html = emperor_html.replace(support_dir + '/', '')

# Ensure the output directory exists
output_dir = "iMSMS_emperor_host_static_html"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the HTML file
output_path = os.path.join(output_dir, "modifiedEmperor.html")
with open(output_path, 'w') as f:
    f.write(emperor_html)

print(f"Emperor visualization saved to {output_path}")
print("- Age binned into 5 categories with colors (blue, green, yellow, orange, red)")
print("- Height binned into 4 categories with shapes (Cone, Sphere, Star, Square)")
print("- Direct THREE.js manipulation for shapes included")
print("- Axes renamed from PC1, PC2, PC3 to Axis 1, Axis 2, Axis 3")
print("- Relative paths for better portability")
print("\nScript completed.")
