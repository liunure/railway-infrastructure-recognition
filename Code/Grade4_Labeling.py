# Grade4_Labeling_MEMORY_EFFICIENT_FIXED.py
# Memory-efficient version - never loads full files

import pandas as pd
import numpy as np
import pickle
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import os
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================
DATA1_PATH = r"C:\Users\Leonor_Almeida\Desktop\LTU\1º Ano\3º Quarter\Industrial AI and eMaintenance - Part I Theories & Concepts\Assignment 4\codes\Data 1"
DATA2_PATH = r"C:\Users\Leonor_Almeida\Desktop\LTU\1º Ano\3º Quarter\Industrial AI and eMaintenance - Part I Theories & Concepts\Assignment 4\codes\Data 2"

class MemoryEfficientLabeler:
    def __init__(self, data1_path, data2_path):
        self.data1_path = data1_path
        self.data2_path = data2_path
        self.infrastructure = None
        self.labeled_segments = []
        
    def load_infrastructure_data(self):
        """Load infrastructure data from Data 1 folder"""
        print("\n" + "="*50)
        print("Loading Infrastructure Data...")
        print("="*50)
        
        # Try different possible filenames
        possible_filenames = {
            "Bridge": ["Bridge.csv", "bridge.csv", "bridges.csv"],
            "RailJoint": ["RailJoint.csv", "railjoint.csv", "joint.csv", "Joint.csv"],
            "Turnout": ["Turnout.csv", "turnout.csv", "TurnOut.csv"]
        }
        
        data_frames = []
        
        for category, filenames in possible_filenames.items():
            for filename in filenames:
                file_path = os.path.join(self.data1_path, filename)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        print(f"✅ Found {category}: {filename}")
                        
                        # Find lat/lon columns
                        lat_col = next((col for col in df.columns if 'lat' in col.lower()), None)
                        lon_col = next((col for col in df.columns if 'lon' in col.lower()), None)
                        
                        if lat_col and lon_col:
                            df = df[[lat_col, lon_col]].copy()
                            df.columns = ['latitude', 'longitude']
                            df = df.dropna()
                            df['type'] = category
                            data_frames.append(df)
                            print(f"   Loaded {len(df)} points")
                            break
                    except Exception as e:
                        print(f"   Error: {e}")
                        continue
        
        if data_frames:
            self.infrastructure = pd.concat(data_frames, ignore_index=True)
            print(f"\n✅ Total infrastructure points: {len(self.infrastructure)}")
            print(self.infrastructure['type'].value_counts())
        else:
            print("❌ No infrastructure data found - using sample points")
            self.infrastructure = pd.DataFrame({
                'latitude': [60.48, 60.49, 60.50, 60.51, 60.52],
                'longitude': [15.42, 15.43, 15.44, 15.45, 15.46],
                'type': ['Bridge', 'RailJoint', 'Turnout', 'Bridge', 'RailJoint']
            })
        
        return self.infrastructure
    
    def find_data2_subfolders(self):
        """Find all subfolders in Data2"""
        if not os.path.exists(self.data2_path):
            print(f"❌ Data2 path not found: {self.data2_path}")
            return []
        
        subfolders = [f for f in os.listdir(self.data2_path) 
                     if os.path.isdir(os.path.join(self.data2_path, f))]
        return sorted(subfolders)
    
    def find_matches_in_folder(self, folder_name):
        """
        Find matches without loading full vibration files
        Returns list of (gps_time, infrastructure_type, speed) for matches
        """
        folder_path = os.path.join(self.data2_path, folder_name)
        
        # Check if all required files exist
        required = ['GPS.latitude.csv', 'GPS.longitude.csv', 'GPS.speed.csv', 'GPS.satellites.csv']
        if not all(os.path.exists(os.path.join(folder_path, f)) for f in required):
            return []
        
        try:
            # Load GPS data (this is small - only ~20K rows max)
            gps_lat = pd.read_csv(os.path.join(folder_path, 'GPS.latitude.csv'), header=None)
            gps_lon = pd.read_csv(os.path.join(folder_path, 'GPS.longitude.csv'), header=None)
            gps_speed = pd.read_csv(os.path.join(folder_path, 'GPS.speed.csv'), header=None)
            gps_sat = pd.read_csv(os.path.join(folder_path, 'GPS.satellites.csv'), header=None)
            
            # Check lengths
            min_len = min(len(gps_lat), len(gps_lon), len(gps_speed), len(gps_sat))
            if min_len < 10:
                return []
            
            # Create GPS dataframe (small)
            gps_df = pd.DataFrame({
                'latitude': gps_lat[0].values[:min_len],
                'longitude': gps_lon[0].values[:min_len],
                'speed': gps_speed[0].values[:min_len],
                'satellites': gps_sat[0].values[:min_len],
                'timestamp': np.arange(min_len) * 0.05
            })
            
            # Filter GPS
            gps_df = gps_df[gps_df['satellites'] >= 4]
            gps_df = gps_df[(gps_df['speed'] > 0.1) & (gps_df['speed'] <= 50)]
            
            if len(gps_df) < 10:
                return []
            
            # Sample GPS points for matching (take every 10th)
            gps_sample = gps_df.iloc[::10]
            
            # Find matches
            matches = []
            for _, infra in self.infrastructure.iterrows():
                infra_coord = (infra['latitude'], infra['longitude'])
                
                for _, gps in gps_sample.iterrows():
                    try:
                        dist = geodesic(infra_coord, (gps['latitude'], gps['longitude'])).meters
                        if dist <= 50:  # Within 50 meters
                            matches.append({
                                'type': infra['type'],
                                'gps_time': gps['timestamp'],
                                'gps_index': int(gps['timestamp'] / 0.05),  # Approximate row index
                                'speed': gps['speed']
                            })
                            break  # One match per infrastructure point
                    except:
                        continue
            
            return matches
            
        except Exception as e:
            print(f"   Error: {e}")
            return []
    
    def extract_vibration_chunk(self, folder_name, gps_time, window_seconds=5):
        """
        Extract ONLY the needed vibration chunk without loading entire file
        """
        folder_path = os.path.join(self.data2_path, folder_name)
        
        # Calculate which rows we need
        samples_per_second = 500
        window_samples = window_seconds * samples_per_second
        center_row = int(gps_time * samples_per_second)
        start_row = max(0, center_row - window_samples // 2)
        
        try:
            # Read only the chunk we need from each vibration file
            vib1_path = os.path.join(folder_path, 'CH1_ACCEL1Z1.csv')
            vib2_path = os.path.join(folder_path, 'CH2_ACCEL1Z2.csv')
            
            if not os.path.exists(vib1_path) or not os.path.exists(vib2_path):
                return None
            
            # Use skiprows and nrows to read only the chunk
            if start_row > 0:
                chunk1 = pd.read_csv(vib1_path, header=None, 
                                    skiprows=range(1, start_row),
                                    nrows=window_samples)
                chunk2 = pd.read_csv(vib2_path, header=None,
                                    skiprows=range(1, start_row),
                                    nrows=window_samples)
            else:
                chunk1 = pd.read_csv(vib1_path, header=None, nrows=window_samples)
                chunk2 = pd.read_csv(vib2_path, header=None, nrows=window_samples)
            
            if len(chunk1) == window_samples and len(chunk2) == window_samples:
                return {
                    'vibration1': chunk1[0].values,
                    'vibration2': chunk2[0].values,
                    'timestamps': np.arange(window_samples) * 0.002
                }
            
        except Exception as e:
            print(f"   Error extracting chunk: {e}")
        
        return None
    
    def collect_samples(self, samples_per_type=10):
        """Collect samples efficiently"""
        print("\n" + "="*60)
        print(f"Collecting up to {samples_per_type} samples per type...")
        print("="*60)
        
        folders = self.find_data2_subfolders()
        print(f"Found {len(folders)} folders to check")
        
        # Shuffle folders to get random sampling
        random.shuffle(folders)
        
        # Track counts
        counts = {'Bridge': 0, 'RailJoint': 0, 'Turnout': 0}
        other_count = 0
        
        # First pass: collect infrastructure matches
        print("\n📌 Phase 1: Collecting infrastructure matches...")
        
        for i, folder in enumerate(folders):
            # Check if we have enough of each type
            if all(counts[t] >= samples_per_type for t in counts):
                print(f"\n✅ Reached target for all infrastructure types!")
                break
            
            print(f"\n[{i+1}/{len(folders)}] Checking {folder[:30]}...")
            
            # Find matches in this folder
            matches = self.find_matches_in_folder(folder)
            
            if matches:
                print(f"   Found {len(matches)} potential matches")
                
                for match in matches:
                    type_name = match['type']
                    
                    if counts[type_name] < samples_per_type:
                        # Extract only the needed chunk
                        segment = self.extract_vibration_chunk(
                            folder, 
                            match['gps_time'],
                            window_seconds=5
                        )
                        
                        if segment:
                            self.labeled_segments.append({
                                'label': type_name,
                                'vibration1': segment['vibration1'],
                                'vibration2': segment['vibration2'],
                                'speed': match['speed'],
                                'folder': folder
                            })
                            counts[type_name] += 1
                            print(f"   ✅ Added {type_name} ({counts[type_name]}/{samples_per_type})")
        
        # Second pass: collect "other" samples
        print(f"\n📌 Phase 2: Collecting 'other' samples...")
        
        # Reshuffle folders
        random.shuffle(folders)
        
        for folder in folders:
            if other_count >= samples_per_type:
                break
            
            # Pick random GPS time from this folder
            try:
                # Just check file exists
                lat_path = os.path.join(self.data2_path, folder, 'GPS.latitude.csv')
                if not os.path.exists(lat_path):
                    continue
                
                # Get file size to estimate rows
                with open(lat_path, 'r') as f:
                    num_rows = sum(1 for _ in f)
                
                if num_rows < 10:
                    continue
                
                # Pick random time
                random_row = random.randint(10, num_rows - 10)
                random_time = random_row * 0.05
                
                segment = self.extract_vibration_chunk(folder, random_time, window_seconds=5)
                if segment:
                    self.labeled_segments.append({
                        'label': 'other',
                        'vibration1': segment['vibration1'],
                        'vibration2': segment['vibration2'],
                        'speed': 0,  # Unknown
                        'folder': folder
                    })
                    other_count += 1
                    print(f"   ✅ Added other ({other_count}/{samples_per_type})")
                    
            except Exception as e:
                continue
        
        return counts, other_count
    
    def save_data(self):
        """Save collected data"""
        print("\n" + "="*60)
        print("Saving collected data...")
        print("="*60)
        
        if not self.labeled_segments:
            print("❌ No data to save")
            return
        
        # Save as pickle
        with open('labeled_segments.pkl', 'wb') as f:
            pickle.dump(self.labeled_segments, f)
        print(f"✅ Saved {len(self.labeled_segments)} segments to 'labeled_segments.pkl'")
        
        # Print summary
        print("\n📊 Final counts:")
        counts = {}
        for seg in self.labeled_segments:
            label = seg['label']
            counts[label] = counts.get(label, 0) + 1
        
        for label, count in counts.items():
            print(f"   {label}: {count}")
    
    def quick_visualize(self):
        """Quick visualization of first few samples"""
        if not self.labeled_segments:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        types = ['Bridge', 'RailJoint', 'Turnout', 'other']
        
        for idx, type_name in enumerate(types):
            ax = axes[idx//2, idx%2]
            type_samples = [s for s in self.labeled_segments if s['label'] == type_name]
            
            if type_samples:
                for i, sample in enumerate(type_samples[:3]):
                    ax.plot(sample['vibration1'][:500], alpha=0.7, linewidth=0.5)
                ax.set_title(f'{type_name} ({len(type_samples)} samples)')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('samples_preview.png', dpi=150)
        plt.show()
        print("✅ Saved 'samples_preview.png'")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("="*70)
    print("GRADE 4 - MEMORY EFFICIENT VERSION")
    print("="*70)
    print("\nThis version:")
    print("✅ Never loads full vibration files")
    print("✅ Reads only the chunks needed")
    print("✅ Uses random sampling to find matches faster")
    print("✅ Stops immediately when targets reached")
    
    # Initialize
    labeler = MemoryEfficientLabeler(DATA1_PATH, DATA2_PATH)
    
    # Load infrastructure
    infra = labeler.load_infrastructure_data()
    
    # Ask for samples per type (smaller is better for memory)
    try:
        samples = int(input("\nHow many samples per type? (5-15 recommended): ") or "10")
        samples = max(3, min(15, samples))
    except:
        samples = 10
    
    print(f"\n🎯 Target: {samples} samples per type")
    
    # Collect samples
    counts, other_count = labeler.collect_samples(samples_per_type=samples)
    
    # Show results
    print("\n" + "="*70)
    print("COLLECTION COMPLETE!")
    print("="*70)
    print(f"Total segments: {len(labeler.labeled_segments)}")
    for type_name, count in counts.items():
        print(f"   {type_name}: {count}")
    print(f"   other: {other_count}")
    
    # Visualize and save
    if labeler.labeled_segments:
        labeler.quick_visualize()
        labeler.save_data()
        print("\n✅ Ready for Grade 5 classification!")
    else:
        print("\n❌ No segments collected")
        print("Try adjusting the distance threshold or checking infrastructure locations")

if __name__ == "__main__":
    main()