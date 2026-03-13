# Grade3_Code1_Infrastructure_Map_ULTRA_SIMPLE.py
# Ultra simple version that just saves files without showing plots

import pandas as pd
import matplotlib.pyplot as plt
import os

# Force matplotlib to use non-interactive backend
plt.switch_backend('Agg')

#############################################################
# Define file paths
base_path = r"C:\Users\Leonor_Almeida\Desktop\LTU\1º Ano\3º Quarter\Industrial AI and eMaintenance - Part I Theories & Concepts\Assignment 4\codes\Data 1"
files = {
    "Bridge": os.path.join(base_path, "Bridge.csv"),
    "RailJoint": os.path.join(base_path, "RailJoint.csv"),  
    "Turnout": os.path.join(base_path, "Turnout.csv")
}
#############################################################

print("Loading infrastructure data...")

# Load and combine data
all_data = []
for category, file in files.items():
    df = pd.read_csv(file)
    # Assume first two columns are lat/lon
    df = df.iloc[:, :2]
    df.columns = ['Latitude', 'Longitude']
    df['Category'] = category
    all_data.append(df)
    print(f"  Loaded {category}: {len(df)} points")

data = pd.concat(all_data, ignore_index=True)

# Save to CSV
data.to_csv('infrastructure_points.csv', index=False)
print(f"\n✅ Saved {len(data)} points to infrastructure_points.csv")

# Create simple plot
plt.figure(figsize=(12, 8))
colors = {'Bridge': 'red', 'RailJoint': 'blue', 'Turnout': 'green'}

for category, color in colors.items():
    subset = data[data['Category'] == category]
    plt.scatter(subset['Longitude'], subset['Latitude'], 
               c=color, label=category, s=50, alpha=0.7)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Railway Infrastructure Map')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('infrastructure_map.png', dpi=150, bbox_inches='tight')
print("✅ Saved infrastructure_map.png")

print("\nDone! Check the files created.")