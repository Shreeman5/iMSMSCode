import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the datasets
S1_PATH = 'iMSMS_dataset/Supplementary_Dataset_S1.xlsx'
S2_PATH = 'iMSMS_dataset/Supplementary_Dataset_S2.xlsx'
S3_PATH = 'iMSMS_dataset/Supplementary_Dataset_S3.xlsx'
S5_PATH = 'iMSMS_dataset/Supplementary_Dataset_S5.xlsx'
S8_PATH = 'iMSMS_dataset/Supplementary_Dataset_S8.xlsx'

sheet1_1 = pd.read_excel(S1_PATH, sheet_name='Dataset S1.1')
sheet1_2 = pd.read_excel(S1_PATH, sheet_name='Dataset S1.2')
sheet2 = pd.read_excel(S2_PATH, sheet_name='Dataset S2')
sheet3 = pd.read_excel(S3_PATH, sheet_name='Dataset S3')
sheet5_1 = pd.read_excel(S5_PATH, sheet_name='Dataset S5.1')
sheet8_3 = pd.read_excel(S8_PATH, sheet_name='Dataset S8.6')


df = (sheet1_2
      .merge(sheet3, on='iMSMS_ID', how='inner')
      .merge(sheet2, on='iMSMS_ID', how='inner')
      .merge(sheet5_1, on='iMSMS_ID', how='inner')
.merge(sheet8_3, on='iMSMS_ID', how='inner')
      )

#df = df[(df["disease"] == "MS")]


#y_params = df[["Methylbutyric.acid","Acetic.acid", "Isovaleric.acid","Hexanoic.acid","Butyric.acid","Isobutyric.acid","Propionic.acid","Valeric.acid"]]
x_params = df[["shannon","chao1",]]
y_params = df[
    ["SFA","SODIUM",'age', "weight", "bmi", 'Alcohol % of cals (%)', 'WHOLEGRAIN', 'Vegetables group (cups)',
     'Dietary Fiber (g)',
     'TOTALVEG', 'TOTALFRUIT',
     "HEI2015_TOTAL_SCORE", "ADDSUG", "MSSS", "edss", "vitamin D (IU)"]]



category_filter = 'site'
category_types = df[category_filter].unique().tolist()
# category_types.append('Overall')

n_treatments = len(category_types)
fig, axes = plt.subplots(n_treatments, 1, figsize=(10, 8), sharex=True, sharey=True)

global_min, global_max = float('inf'), float('-inf')
correlation_matrices = {}


label_mapping = {
    # "": "",
    "0.0": "Don't have pets",
    "1.0": "Have Pets",
    "RRMS": "Relapsing-Remitting MS (RRMS)",
    "SPMS": "Secondary Progressive MS (SPMS)",
    "PPMS": "Primary Progressive MS (PPMS)",
    "PRMS": "Progressive-Relapsing MS (PRMS)",
    "PMS": "Progressive MS (PMS)",
    "Control_RRMS": "HHC of RRMS",
    "Control_PMS": "HHC of PMS",
    "MS": "with MS Condition",
    "Control": "HHC (Household Healthy Controls)",
    "smoker": "Smoker",
    "nonsmoker": "Non Smoker",
    "formersmoker": "Former Smoker",
    "F": "Female",
    "M": "Male",
    "weight": "Weight",
    "height": "Height",
    "age": "Age",
    "bmi": "BMI",
    "shannon": "Shannon Index",
    "chao1": "Chao1 Index",
    "ADDSUG": "Added Sugars",
    "Vegetables group (cups)": "Vegetable\n Intake (Cups)",
    "Dietary Fiber (g)": "Dietary Fiber (grams)",
    "TOTALVEG": "Total Vegetables\n(Eating Index Score)",
    "TOTALFRUIT": "Total Fruits\n(Eating Index Score)",
    "HEI2015_TOTAL_SCORE": "Total Eating Index Score",
    "edss": "Expanded Disability\nStatus Scale",
    "MSSS": "Multiple Sclerosis\nSeverity Score",
    "vitamin D (IU)": "Vitamin D (IU)",
}

from scipy.stats import pearsonr, spearmanr





for category in category_types:
    treated_df = df if category == 'Overall' else df[(df[category_filter] == category)]

    correlation_matrix = pd.DataFrame(index=x_params.columns, columns=y_params.columns)

    for x_col in x_params.columns:
        for y_col in y_params.columns:

            valid_data = treated_df[[x_col, y_col]].dropna()
            if len(valid_data) > 1:

                correlation_value, _ = spearmanr(valid_data[x_col], valid_data[y_col])

            else:
                correlation_value = None

            correlation_matrix.loc[x_col, y_col] = correlation_value


            global_min = min(global_min, correlation_value) if correlation_value is not None else global_min
            global_max = max(global_max, correlation_value) if correlation_value is not None else global_max

    correlation_matrices[category] = correlation_matrix


for ax, category in zip(axes, category_types):
    treated_df = df if category == 'Overall' else df[(df[category_filter] == category)]
    correlation_matrix = correlation_matrices[category]


    num_items = treated_df.shape[0]

    sns.heatmap(correlation_matrix.astype(float), annot=True, fmt=".2f", cmap='seismic', square=False,
                cbar_kws={"shrink": 0.65}, linewidths=1, ax=ax, annot_kws={"size": 8, "weight": "bold"},
                vmin=-1, vmax=1)

    ax.set_title(f'{label_mapping.get(category, category)} (n={num_items})')
    ax.tick_params(axis='both', which='both', length=0)


    ax.set_yticklabels([label_mapping.get(label, label) for label in correlation_matrix.index], rotation=0, fontsize=10)
    ax.set_xticklabels([label_mapping.get(label, label) for label in correlation_matrix.columns], rotation=90,
                       fontsize=10)

plt.suptitle('Correlation', fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 1))
plt.show()
