import pandas as pd
import os
import glob

# Set the folder containing your CSV files
folder_path = 'D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\similarity_csv'

# Use glob to get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Read and concatenate all CSVs
combined_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Save to a new CSV file
combined_df.to_csv('combined_output.csv', index=False)