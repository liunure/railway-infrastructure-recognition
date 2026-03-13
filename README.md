# Railway Infrastructure Recognition Using Vibration Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project develops an AI-driven approach for automated railway infrastructure monitoring using vibration analysis. It detects three types of infrastructure: **bridges, rail joints, and turnouts** from train-mounted sensor data.

**Course:** Industrial AI and eMaintenance - Part I  
**Institution:** Luleå University of Technology  
**Author:** Leonor Almeida

## How to Run

### Step 1: Install Requirements
pip install pandas numpy scipy scikit-learn tensorflow matplotlib plotly geopy tqdm joblib

### Step 2: Run Grade 3 - Mapping
python code/grade3_infrastructure_map.py
python code/grade3_gps_track.py

This creates maps of infrastructure points and GPS tracks.

### Step 3: Run Grade 4 - Labeling
python code/grade4_labeling.py

This extracts and labels vibration segments.

### Step 4: Run Grade 5 - Classification
python code/grade5_classification.py

This trains ML/DL models and compares performance.

## Key Results

### GPS Data Quality
- **252,000** original GPS points
- **119,643** valid points after filtering
- **52.5%** of points removed due to poor quality

### Labeled Dataset
| Class | Samples |
|-------|---------|
| Bridge | 10 |
| Turnout | 10 |
| Rail Joint | 4 |
| Other | 10 |
| **Total** | **34** |

### Model Performance
| Model | Accuracy |
|-------|----------|
| Extra Trees | 42.9% |
| Naive Bayes | 42.9% |
| Gradient Boosting | 28.6% |
| Random Forest | 28.6% |
| LSTM | 28.6% |
| CNN | 14.3% |

**Best Model: Extra Trees (42.9% accuracy)**

## Results Preview

### Infrastructure Map
![Infrastructure Map](https://results/maps/infrastructure_map.png)
*Red: Bridges, Blue: Rail Joints, Green: Turnouts*

### GPS Tracks
![GPS Tracks](https://results/maps/all_gps_tracks.png)
*Multiple train journeys near Borlänge*

### Sample Vibrations
![Sample Vibrations](https://results/labeling/samples_preview.png)
*Vibration patterns for each infrastructure type*

### Model Comparison
![Model Comparison](https://results/classification/final_model_comparison.png)
*Performance comparison of all models*

## Technical Details

### Data Sources
- **Infrastructure data**: 120 points
  - 75 turnouts
  - 25 bridges
  - 20 rail joints
- **Sensor data**: 139 train journeys with:
  - Vibration sensors: 500 Hz sampling rate
  - GPS: 20 Hz sampling rate
  - Speed and satellite count

### Features Extracted (93 total)
- **Time-domain**: mean, std, RMS, skewness, kurtosis, percentiles
- **Frequency-domain**: spectral centroid, band energies, dominant frequency
- **Cross-channel**: correlation, phase difference, coherence