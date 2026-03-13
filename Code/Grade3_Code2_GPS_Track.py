# Grade3_Code2_GPS_Track_MULTIFOLDER.py
# Handles multiple subfolders in Data2

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================
DATA1_PATH = r"C:\Users\Leonor_Almeida\Desktop\LTU\1º Ano\3º Quarter\Industrial AI and eMaintenance - Part I Theories & Concepts\Assignment 4\codes\Data 1"
DATA2_PATH = r"C:\Users\Leonor_Almeida\Desktop\LTU\1º Ano\3º Quarter\Industrial AI and eMaintenance - Part I Theories & Concepts\Assignment 4\codes\Data 2"

# ============================================
# PART 1: Load Infrastructure Data (from Data 1)
# ============================================

def load_infrastructure_data():
    """Load infrastructure data from Data 1 folder"""
    print("\n" + "="*60)
    print("Loading Infrastructure Data from Data 1...")
    print("="*60)
    
    files = {
        "Bridge": os.path.join(DATA1_PATH, "Bridge.csv"),
        "RailJoint": os.path.join(DATA1_PATH, "RailJoint.csv"),
        "Turnout": os.path.join(DATA1_PATH, "Turnout.csv")
    }
    
    infra_data = []
    for category, file_path in files.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding="utf-8")
                df.columns = df.columns.str.strip()
                
                # Find latitude/longitude columns
                lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
                lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
                
                if lat_col and lon_col:
                    df["Latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
                    df["Longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
                    df = df[["Latitude", "Longitude"]].dropna()
                    df["Category"] = category
                    infra_data.append(df)
                    print(f"✅ Loaded {category}: {len(df)} points")
                else:
                    print(f"⚠️ Could not find coordinates in {category}")
            else:
                print(f"⚠️ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Error loading {category}: {e}")
    
    if infra_data:
        infra_df = pd.concat(infra_data, ignore_index=True)
        print(f"\n📊 Total infrastructure points: {len(infra_df)}")
        print(infra_df['Category'].value_counts())
        return infra_df
    else:
        print("⚠️ No infrastructure data loaded")
        return pd.DataFrame()

# ============================================
# PART 2: Find all Data2 subfolders
# ============================================

def find_data2_subfolders():
    """Find all subfolders in Data2 that contain the required files"""
    print("\n" + "="*60)
    print("Scanning Data2 subfolders...")
    print("="*60)
    
    if not os.path.exists(DATA2_PATH):
        print(f"❌ ERROR: Data2 folder not found at {DATA2_PATH}")
        return []
    
    # Get all subfolders
    subfolders = [f for f in os.listdir(DATA2_PATH) 
                  if os.path.isdir(os.path.join(DATA2_PATH, f))]
    
    print(f"Found {len(subfolders)} subfolders in Data2")
    
    # Check each subfolder for required files
    valid_folders = []
    required_files = [
        'CH1_ACCEL1Z1.csv',
        'CH2_ACCEL1Z2.csv',
        'GPS.latitude.csv',
        'GPS.longitude.csv',
        'GPS.speed.csv',
        'GPS.satellites.csv'
    ]
    
    for folder in sorted(subfolders):
        folder_path = os.path.join(DATA2_PATH, folder)
        files_present = os.listdir(folder_path)
        
        # Check if all required files exist
        missing = [f for f in required_files if f not in files_present]
        
        if not missing:
            valid_folders.append(folder)
            print(f"  ✅ {folder} - Complete")
        else:
            print(f"  ⚠️ {folder} - Missing: {missing}")
    
    print(f"\n📊 Found {len(valid_folders)} valid subfolders with complete data")
    return valid_folders

# ============================================
# PART 3: Load data from a specific subfolder
# ============================================

def load_subfolder_data(folder_name):
    """Load GPS and vibration data from a specific subfolder"""
    folder_path = os.path.join(DATA2_PATH, folder_name)
    
    dataframes = {}
    
    # GPS files
    gps_files = {
        'latitude': 'GPS.latitude.csv',
        'longitude': 'GPS.longitude.csv',
        'speed': 'GPS.speed.csv',
        'satellites': 'GPS.satellites.csv'
    }
    
    for key, filename in gps_files.items():
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(filepath, header=None, names=[key])
            df['timestamp'] = df.index * 0.05  # 20 Hz GPS
            dataframes[key] = df
        except Exception as e:
            print(f"    Error loading {filename}: {e}")
            return None, None
    
    # Vibration files
    vib_files = {
        'channel1': 'CH1_ACCEL1Z1.csv',
        'channel2': 'CH2_ACCEL1Z2.csv'
    }
    
    for key, filename in vib_files.items():
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(filepath, header=None, names=[key])
            df['timestamp'] = df.index * 0.002  # 500 Hz vibration
            dataframes[key] = df
        except Exception as e:
            print(f"    Error loading {filename}: {e}")
            return None, None
    
    # Create GPS DataFrame
    gps_df = pd.DataFrame({
        'timestamp': dataframes['latitude']['timestamp'],
        'latitude': dataframes['latitude']['latitude'],
        'longitude': dataframes['longitude']['longitude'],
        'speed': dataframes['speed']['speed'],
        'satellites': dataframes['satellites']['satellites'],
        'folder': folder_name  # Add folder name as column
    })
    
    # Create Vibration DataFrame
    vib_df = pd.DataFrame({
        'timestamp': dataframes['channel1']['timestamp'],
        'channel1': dataframes['channel1']['channel1'],
        'channel2': dataframes['channel2']['channel2'],
        'folder': folder_name  # Add folder name as column
    })
    
    return gps_df, vib_df

# ============================================
# PART 4: Load all data from all valid subfolders
# ============================================

def load_all_sensor_data(valid_folders, max_folders=None):
    """Load data from multiple subfolders"""
    print("\n" + "="*60)
    print("Loading Sensor Data from all valid subfolders...")
    print("="*60)
    
    if max_folders:
        print(f"Loading first {max_folders} folders (you can adjust max_folders parameter)")
        valid_folders = valid_folders[:max_folders]
    
    all_gps = []
    all_vib = []
    
    for i, folder in enumerate(valid_folders):
        print(f"\n[{i+1}/{len(valid_folders)}] Loading {folder}...")
        gps_df, vib_df = load_subfolder_data(folder)
        
        if gps_df is not None and vib_df is not None:
            all_gps.append(gps_df)
            all_vib.append(vib_df)
            print(f"    ✅ Loaded: {len(gps_df)} GPS points, {len(vib_df)} vibration samples")
    
    if all_gps:
        combined_gps = pd.concat(all_gps, ignore_index=True)
        combined_vib = pd.concat(all_vib, ignore_index=True)
        
        print(f"\n📊 TOTAL DATA LOADED:")
        print(f"   GPS points: {len(combined_gps)} from {len(all_gps)} journeys")
        print(f"   Vibration samples: {len(combined_vib)} from {len(all_vib)} journeys")
        print(f"   Time range: {combined_gps['timestamp'].min():.1f}s to {combined_gps['timestamp'].max():.1f}s")
        
        return combined_gps, combined_vib
    else:
        print("❌ No data loaded")
        return pd.DataFrame(), pd.DataFrame()

# ============================================
# PART 5: Filter and Clean GPS Data
# ============================================

def filter_gps_data(gps_df, min_satellites=4, max_speed=50):
    """Filter out inaccurate GPS points"""
    print("\n" + "="*60)
    print("Filtering GPS Data...")
    print("="*60)
    
    if gps_df.empty:
        print("❌ No GPS data to filter")
        return gps_df
    
    original_count = len(gps_df)
    
    # Filter by satellite count
    gps_filtered = gps_df[gps_df['satellites'] >= min_satellites].copy()
    
    # Filter by realistic speed
    gps_filtered = gps_filtered[(gps_filtered['speed'] > 0.1) & (gps_filtered['speed'] <= max_speed)]
    
    print(f"   Original points: {original_count}")
    print(f"   After filtering: {len(gps_filtered)}")
    print(f"   Removed: {original_count - len(gps_filtered)} points ({((original_count - len(gps_filtered))/original_count*100):.1f}%)")
    
    # Show stats by folder
    if 'folder' in gps_filtered.columns:
        print("\n   Points by journey after filtering:")
        folder_stats = gps_filtered['folder'].value_counts()
        for folder, count in folder_stats.items():
            print(f"     {folder}: {count} points")
    
    return gps_filtered

# ============================================
# PART 6: Create Maps
# ============================================

def create_maps(infra_df, gps_df):
    """Create various maps"""
    print("\n" + "="*60)
    print("Creating Maps...")
    print("="*60)
    
    # Define infrastructure colors ONCE at the beginning
    infra_colors = {'Bridge': 'red', 'RailJoint': 'blue', 'Turnout': 'green'}
    
    # Map 1: Infrastructure only
    plt.figure(figsize=(14, 10))
    for category, color in infra_colors.items():
        subset = infra_df[infra_df['Category'] == category]
        if len(subset) > 0:
            plt.scatter(subset['Longitude'], subset['Latitude'], 
                       c=color, label=category, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Railway Infrastructure Map - All Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('infrastructure_all_points.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved 'infrastructure_all_points.png'")
    
    if gps_df.empty:
        print("⚠️ No GPS data for track maps")
        return
    
    # Map 2: All GPS tracks - USE A DIFFERENT VARIABLE NAME for track colors
    plt.figure(figsize=(16, 12))
    
    # Plot each journey with different colors
    if 'folder' in gps_df.columns:
        unique_folders = gps_df['folder'].unique()
        track_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_folders)))  # DIFFERENT NAME!
        
        for i, folder in enumerate(unique_folders):
            subset = gps_df[gps_df['folder'] == folder]
            plt.plot(subset['longitude'], subset['latitude'], 
                    color=track_colors[i], linewidth=1, alpha=0.7, label=folder[:15])
    else:
        plt.plot(gps_df['longitude'], gps_df['latitude'], 
                'b-', linewidth=1, alpha=0.7)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('All GPS Tracks')
    if 'folder' in gps_df.columns:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_gps_tracks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved 'all_gps_tracks.png'")
    
    # Map 3: Combined - USE THE ORIGINAL infra_colors
    plt.figure(figsize=(16, 12))
    
    # Plot first journey track
    if 'folder' in gps_df.columns:
        first_folder = gps_df['folder'].iloc[0]
        first_journey = gps_df[gps_df['folder'] == first_folder]
        plt.plot(first_journey['longitude'], first_journey['latitude'], 
                'gray', linewidth=2, alpha=0.8, label=f'Track: {first_folder[:20]}')
    
    # Plot infrastructure - NOW USING infra_colors (still a dictionary!)
    for category, color in infra_colors.items():  # <-- FIXED: using infra_colors
        subset = infra_df[infra_df['Category'] == category]
        if len(subset) > 0:
            plt.scatter(subset['Longitude'], subset['Latitude'], 
                       c=color, label=category, s=150, alpha=0.8, 
                       edgecolors='black', linewidth=1)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample Journey with Infrastructure Overlay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sample_journey_with_infrastructure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved 'sample_journey_with_infrastructure.png'")

# ============================================
# PART 7: Create Journey Analysis
# ============================================

def create_journey_analysis(gps_df, vib_df):
    """Create analysis plots for each journey"""
    print("\n" + "="*60)
    print("Creating Journey Analysis...")
    print("="*60)
    
    if gps_df.empty:
        print("❌ No GPS data for analysis")
        return
    
    # Analyze by folder
    if 'folder' in gps_df.columns:
        folders = gps_df['folder'].unique()
        
        # Summary statistics
        summary = []
        for folder in folders:
            gps_subset = gps_df[gps_df['folder'] == folder]
            
            # Calculate journey stats
            duration = gps_subset['timestamp'].max() - gps_subset['timestamp'].min()
            avg_speed = gps_subset['speed'].mean()
            max_speed = gps_subset['speed'].max()
            
            # Calculate distance
            distance = 0
            for i in range(1, len(gps_subset)):
                try:
                    dist = geodesic(
                        (gps_subset.iloc[i-1]['latitude'], gps_subset.iloc[i-1]['longitude']),
                        (gps_subset.iloc[i]['latitude'], gps_subset.iloc[i]['longitude'])
                    ).meters
                    distance += dist
                except:
                    pass
            
            summary.append({
                'Journey': folder,
                'GPS Points': len(gps_subset),
                'Duration (s)': round(duration, 1),
                'Distance (km)': round(distance/1000, 2),
                'Avg Speed (km/h)': round(avg_speed * 3.6, 1),
                'Max Speed (km/h)': round(max_speed * 3.6, 1)
            })
        
        summary_df = pd.DataFrame(summary)
        print("\n📊 Journey Summary:")
        print(summary_df.to_string(index=False))
        summary_df.to_csv('journey_summary.csv', index=False)
        print("✅ Saved 'journey_summary.csv'")
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Duration comparison
        axes[0, 0].bar(range(len(summary_df)), summary_df['Duration (s)'])
        axes[0, 0].set_xticks(range(len(summary_df)))
        axes[0, 0].set_xticklabels([f"J{i+1}" for i in range(len(summary_df))], rotation=45)
        axes[0, 0].set_title('Journey Duration')
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distance comparison
        axes[0, 1].bar(range(len(summary_df)), summary_df['Distance (km)'])
        axes[0, 1].set_xticks(range(len(summary_df)))
        axes[0, 1].set_xticklabels([f"J{i+1}" for i in range(len(summary_df))], rotation=45)
        axes[0, 1].set_title('Journey Distance')
        axes[0, 1].set_ylabel('Kilometers')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Speed comparison
        axes[1, 0].bar(range(len(summary_df)), summary_df['Avg Speed (km/h)'], 
                      alpha=0.7, label='Avg')
        axes[1, 0].bar(range(len(summary_df)), summary_df['Max Speed (km/h)'], 
                      alpha=0.7, label='Max', bottom=summary_df['Avg Speed (km/h)'])
        axes[1, 0].set_xticks(range(len(summary_df)))
        axes[1, 0].set_xticklabels([f"J{i+1}" for i in range(len(summary_df))], rotation=45)
        axes[1, 0].set_title('Journey Speeds')
        axes[1, 0].set_ylabel('km/h')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Points count
        axes[1, 1].bar(range(len(summary_df)), summary_df['GPS Points'])
        axes[1, 1].set_xticks(range(len(summary_df)))
        axes[1, 1].set_xticklabels([f"J{i+1}" for i in range(len(summary_df))], rotation=45)
        axes[1, 1].set_title('GPS Points per Journey')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('journey_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved 'journey_comparison.png'")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("="*70)
    print("GRADE 3 - MAPPING: Multiple Journeys Analysis")
    print("="*70)
    print(f"\nData1 path: {DATA1_PATH}")
    print(f"Data2 path: {DATA2_PATH}")
    
    # Check if paths exist
    if not os.path.exists(DATA1_PATH):
        print(f"\n❌ ERROR: Data1 folder not found")
        print("Please update DATA1_PATH variable with the correct path")
        return
    
    if not os.path.exists(DATA2_PATH):
        print(f"\n❌ ERROR: Data2 folder not found")
        print("Please update DATA2_PATH variable with the correct path")
        return
    
    # Step 1: Load infrastructure data
    infra_df = load_infrastructure_data()
    
    # Step 2: Find all valid subfolders
    valid_folders = find_data2_subfolders()
    
    if not valid_folders:
        print("\n❌ No valid subfolders found. Check Data2 contents.")
        return
    
    # Step 3: Ask user how many folders to process
    print(f"\nFound {len(valid_folders)} valid journeys.")
    print("You can process all or select a subset (to save time/memory).")
    
    try:
        max_folders = input("How many journeys to process? (press Enter for all): ").strip()
        if max_folders:
            max_folders = int(max_folders)
        else:
            max_folders = None
    except:
        max_folders = None
        print("Using all folders")
    
    # Step 4: Load data from selected folders
    gps_df, vib_df = load_all_sensor_data(valid_folders, max_folders)
    
    if gps_df.empty:
        print("\n❌ No data loaded. Exiting.")
        return
    
    # Step 5: Filter GPS data
    gps_filtered = filter_gps_data(gps_df)
    
    # Step 6: Create maps
    create_maps(infra_df, gps_filtered)
    
    # Step 7: Create journey analysis
    create_journey_analysis(gps_filtered, vib_df)
    
    # Step 8: Save processed data
    print("\n" + "="*60)
    print("Saving Processed Data...")
    print("="*60)
    
    gps_filtered.to_csv('all_filtered_gps.csv', index=False)
    print("✅ Saved 'all_filtered_gps.csv'")
    
    # Save infrastructure data for Grade 4
    infra_df.to_csv('infrastructure_for_labeling.csv', index=False)
    print("✅ Saved 'infrastructure_for_labeling.csv'")
    
    print("\n" + "="*70)
    print("✅ GRADE 3 - CODE 2 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFiles created:")
    print("   📁 infrastructure_all_points.png")
    print("   📁 all_gps_tracks.png")
    print("   📁 sample_journey_with_infrastructure.png")
    print("   📁 journey_comparison.png")
    print("   📁 journey_summary.csv")
    print("   📁 all_filtered_gps.csv")
    print("   📁 infrastructure_for_labeling.csv")

if __name__ == "__main__":
    main()