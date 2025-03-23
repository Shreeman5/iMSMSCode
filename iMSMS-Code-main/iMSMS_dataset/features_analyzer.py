import pandas as pd
import numpy as np
import os

def analyze_csv(file_path):
    """
    Analyzes a CSV file to identify categorical and numerical columns,
    and provides summary information for each type.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        None: Prints analysis results
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV file: {file_path}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Total rows: {len(df)}")
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Analyze each column
    for column in df.columns:
        print(f"Column: {column}")
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Handle numerical column
            min_val = df[column].min()
            max_val = df[column].max()
            print(f"  Type: Numerical")
            print(f"  Range: {min_val} to {max_val}")
            print(f"  Mean: {df[column].mean():.2f}")
            print(f"  Median: {df[column].median()}")
            print(f"  Number of null values: {df[column].isna().sum()}")
        else:
            # Handle categorical column
            unique_values = df[column].unique()
            print(f"  Type: Categorical")
            print(f"  Number of unique values: {len(unique_values)}")
            print(f"  Number of null values: {df[column].isna().sum()}")
            
            # Print unique values (limiting to 20 for readability)
            if len(unique_values) <= 20:
                print(f"  Unique values: {', '.join(str(val) for val in unique_values)}")
            else:
                sample_values = np.random.choice(unique_values, size=10, replace=False)
                print(f"  Sample of unique values (10 of {len(unique_values)}): {', '.join(str(val) for val in sample_values)}")
        
        print("\n" + "-"*30 + "\n")

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with the path to your CSV file
    csv_path = "/home/shreeman/Desktop/CodeForMSTool/iMSMS-Code-main/iMSMS_dataset/csvs/Supplementary_Dataset_S5.1.csv"
    analyze_csv(csv_path)