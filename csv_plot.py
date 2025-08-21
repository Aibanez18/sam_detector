import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from random import sample

# Settings
folder_path = 'D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\similarity_csv'
file_filter = lambda name: name.endswith('.csv')  # Modify if you want a more specific filter

# Get CSV files (you can customize how you choose which files to use)
csv_files = [f for f in os.listdir(folder_path) if file_filter(f)]
selected_files = sample(csv_files, 9)  # Take only 9 files

# Set up 3x3 plot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, file_name in enumerate(selected_files):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    ax = axes[i]
    
    # Filter labels
    grouped = df.groupby('label')
    
    for label, group in grouped:
        sims = df[df['label'] == label]['similarity'].dropna().values
        if len(sims) == 0:
            continue
        
        mu = np.mean(sims)
        sigma = np.std(sims)

        # Plot PDF of normal distribution
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, label=f"{label} (μ={mu:.2f}, σ={sigma:.2f})")
    
    ax.set_title(f"{file_name}")
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Probability Density")
    ax.legend()

# Global layout
plt.tight_layout()
plt.savefig('sim_dist.png')
plt.show()
