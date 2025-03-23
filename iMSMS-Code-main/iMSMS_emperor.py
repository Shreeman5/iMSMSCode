import pandas as pd
import numpy as np
from emperor import Emperor
from skbio import OrdinationResults
from emperor.util import get_emperor_support_files_dir
from skbio.stats.ordination import pcoa

from scipy.spatial.distance import pdist, squareform

# Function that converts integer columns to range values
def convert_to_ranges(df, num_bins=10):

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
        range_labels = [f"{round(range_steps[i], 2)}-{round(range_steps[i + 1], 2)}" for i in
                        range(len(range_steps) - 1)]


        df[col] = df[col].apply(categorize_value, args=(min_val, max_val, range_steps, range_labels))

    return df

S1_PATH = 'iMSMS_dataset/Supplementary_Dataset_S1.xlsx'
S2_PATH = 'iMSMS_dataset/Supplementary_Dataset_S2.xlsx'
S3_PATH = 'iMSMS_dataset/Supplementary_Dataset_S3.xlsx'
S5_PATH = 'iMSMS_dataset/Supplementary_Dataset_S5.xlsx'
S6_PATH = 'iMSMS_dataset/Supplementary_Dataset_S6.xlsx'

sheet1_2 = pd.read_excel(S1_PATH, sheet_name='Dataset S1.2')
sheet2 = pd.read_excel(S2_PATH, sheet_name='Dataset S2')
sheet3 = pd.read_excel(S3_PATH, sheet_name='Dataset S3')


sheet6_class = pd.read_excel(S6_PATH, sheet_name='class')
demographic_data = (sheet1_2
      .merge(sheet3, on='iMSMS_ID', how='inner')
      .merge(sheet2, on='iMSMS_ID', how='inner')

      )

demographic_data = convert_to_ranges(demographic_data, num_bins=5)


sheet6_class = sheet6_class.merge(demographic_data[['iMSMS_ID']], on='iMSMS_ID', how='inner')

demographic_data = demographic_data.set_index('iMSMS_ID')
sheet6_class = sheet6_class.set_index('iMSMS_ID')


#Beta Diversity
bray_curtis = pdist(sheet6_class, metric='braycurtis')

bray_curtis_matrix = squareform(bray_curtis)

bray_curtis_df = pd.DataFrame(bray_curtis_matrix, index=sheet6_class.index, columns=sheet6_class.index)

site_names = demographic_data.index.tolist()

num_samples = len(demographic_data)

distance_matrix = bray_curtis_df.to_numpy()

distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix, 0)


pcoa_results = pcoa(distance_matrix)


eigenvalues = np.clip(pcoa_results.eigvals, a_min=0, a_max=None)

proportion_explained = eigenvalues / eigenvalues.sum()

site_coordinates = pcoa_results.samples
eigvals = eigenvalues.tolist()
proportion_str = "\t".join(map(str, proportion_explained.tolist()))


site_data = site_coordinates


eigvals_str = "\t".join(map(str, eigvals))
if len(site_names) != num_samples:
    raise ValueError(f"Number of site names ({len(site_names)}) does not match the number of samples ({num_samples})")


site_coords_str = "\n".join([f"{site_names[i]}\t" + "\t".join(map(str, site_data.iloc[i].values)) for i in range(num_samples)])


#Writing ordination data manually
ordination_output = f"""Eigvals\t{len(eigvals)}
{eigvals_str}

Proportion explained\t{len(proportion_explained)}
{proportion_str}

Species\t0\t0

Site\t{len(site_names)}\t{len(site_names)}
{site_coords_str}

Biplot\t0\t0

Site constraints\t0\t0
"""


with open("iMSMS_generated_ordination_data.txt", "w") as f:
    f.write(ordination_output)


res = OrdinationResults.read('iMSMS_generated_ordination_data.txt')
print(res)
viz = Emperor(res, demographic_data,
              #remote=False
              remote=get_emperor_support_files_dir()
              )

#iMSMS-emperor.html will be static html page of emperor with iMSMS data.
with open('iMSMS-emperor.html', 'w') as f:
    f.write(viz.make_emperor(standalone=True))











