# Grade5_Classification.py
# Feature extraction and ML/DL classification

import numpy as np
import pandas as pd
import pickle
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_curve, auc)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================
# PART 1: FEATURE EXTRACTION
# ============================================

class FeatureExtractor:
    """Extract numerical features from vibration signals"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        
    def extract_statistical_features(self, signal_data, channel_name):
        """Extract time-domain statistical features"""
        features = {}
        
        # Basic statistics
        features[f'{channel_name}_mean'] = np.mean(signal_data)
        features[f'{channel_name}_std'] = np.std(signal_data)
        features[f'{channel_name}_var'] = np.var(signal_data)
        features[f'{channel_name}_max'] = np.max(signal_data)
        features[f'{channel_name}_min'] = np.min(signal_data)
        features[f'{channel_name}_peak_to_peak'] = np.ptp(signal_data)
        features[f'{channel_name}_rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Shape statistics
        features[f'{channel_name}_skewness'] = stats.skew(signal_data)
        features[f'{channel_name}_kurtosis'] = stats.kurtosis(signal_data)
        
        # Percentiles
        for p in [10, 25, 50, 75, 90, 95]:
            features[f'{channel_name}_percentile_{p}'] = np.percentile(signal_data, p)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features[f'{channel_name}_zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
        # Signal energy
        features[f'{channel_name}_energy'] = np.sum(signal_data**2)
        
        # Mean absolute deviation
        features[f'{channel_name}_mad'] = np.mean(np.abs(signal_data - np.mean(signal_data)))
        
        # Crest factor (peak/RMS)
        if features[f'{channel_name}_rms'] > 0:
            features[f'{channel_name}_crest_factor'] = np.max(np.abs(signal_data)) / features[f'{channel_name}_rms']
        else:
            features[f'{channel_name}_crest_factor'] = 0
            
        # Shape factor (RMS/mean absolute value)
        if features[f'{channel_name}_mad'] > 0:
            features[f'{channel_name}_shape_factor'] = features[f'{channel_name}_rms'] / features[f'{channel_name}_mad']
        else:
            features[f'{channel_name}_shape_factor'] = 0
        
        # Impulse factor (peak/MAD)
        if features[f'{channel_name}_mad'] > 0:
            features[f'{channel_name}_impulse_factor'] = np.max(np.abs(signal_data)) / features[f'{channel_name}_mad']
        else:
            features[f'{channel_name}_impulse_factor'] = 0
        
        return features
    
    def extract_frequency_features(self, signal_data, channel_name):
        """Extract frequency-domain features using FFT"""
        features = {}
        
        # Compute FFT
        n = len(signal_data)
        fft_vals = fft(signal_data)
        fft_abs = np.abs(fft_vals[:n//2])
        freqs = fftfreq(n, 1/self.sampling_rate)[:n//2]
        
        # Remove DC component
        fft_abs = fft_abs[1:]
        freqs = freqs[1:]
        
        if len(fft_abs) > 0 and np.sum(fft_abs) > 0:
            # Spectral centroid (weighted mean of frequencies)
            features[f'{channel_name}_spectral_centroid'] = np.sum(freqs * fft_abs) / np.sum(fft_abs)
            
            # Spectral spread (variance around centroid)
            spread = np.sqrt(np.sum(((freqs - features[f'{channel_name}_spectral_centroid'])**2) * fft_abs) / np.sum(fft_abs))
            features[f'{channel_name}_spectral_spread'] = spread
            
            # Spectral roll-off (frequency where 85% of energy is below)
            cumsum = np.cumsum(fft_abs)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                features[f'{channel_name}_spectral_rolloff'] = freqs[rolloff_idx[0]]
            else:
                features[f'{channel_name}_spectral_rolloff'] = 0
                
            # Spectral flatness (geometric mean / arithmetic mean)
            geometric_mean = np.exp(np.mean(np.log(fft_abs + 1e-12)))
            arithmetic_mean = np.mean(fft_abs)
            features[f'{channel_name}_spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        else:
            features[f'{channel_name}_spectral_centroid'] = 0
            features[f'{channel_name}_spectral_spread'] = 0
            features[f'{channel_name}_spectral_rolloff'] = 0
            features[f'{channel_name}_spectral_flatness'] = 0
        
        # Band energies (frequency bands)
        bands = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200), 
                 (200, 250), (250, 300), (300, 350), (350, 400), (400, 450), (450, 500)]
        
        for i, (low, high) in enumerate(bands):
            band_mask = (freqs >= low) & (freqs < high)
            if np.sum(band_mask) > 0:
                features[f'{channel_name}_band_{i}_energy'] = np.sum(fft_abs[band_mask]**2)
            else:
                features[f'{channel_name}_band_{i}_energy'] = 0
        
        # Dominant frequency
        if len(fft_abs) > 0:
            peak_idx = np.argmax(fft_abs)
            features[f'{channel_name}_dominant_freq'] = freqs[peak_idx]
            features[f'{channel_name}_dominant_freq_magnitude'] = fft_abs[peak_idx]
        else:
            features[f'{channel_name}_dominant_freq'] = 0
            features[f'{channel_name}_dominant_freq_magnitude'] = 0
        
        # Spectral entropy
        pdf = fft_abs / (np.sum(fft_abs) + 1e-12)
        features[f'{channel_name}_spectral_entropy'] = -np.sum(pdf * np.log2(pdf + 1e-12))
        
        return features
    
    def extract_cross_channel_features(self, signal1, signal2):
        """Extract features comparing both channels"""
        features = {}
        
        # Correlation between channels
        features['cross_correlation'] = np.corrcoef(signal1, signal2)[0, 1]
        
        # Channel ratio features
        rms1 = np.sqrt(np.mean(signal1**2))
        rms2 = np.sqrt(np.mean(signal2**2))
        features['channel_ratio_rms'] = rms1 / (rms2 + 1e-12)
        
        peak1 = np.max(np.abs(signal1))
        peak2 = np.max(np.abs(signal2))
        features['channel_ratio_peak'] = peak1 / (peak2 + 1e-12)
        
        # Phase difference (using cross-correlation lag)
        correlation = np.correlate(signal1 - np.mean(signal1), 
                                  signal2 - np.mean(signal2), mode='same')
        lag = np.argmax(correlation) - len(signal1)//2
        features['phase_lag_samples'] = lag
        features['phase_lag_time'] = lag / self.sampling_rate
        
        # Coherence in frequency bands
        if len(signal1) >= 256 and len(signal2) >= 256:
            try:
                f, coh = signal.coherence(signal1, signal2, fs=self.sampling_rate, 
                                         nperseg=min(256, len(signal1)))
                features['mean_coherence'] = np.mean(coh)
                features['max_coherence'] = np.max(coh)
            except:
                features['mean_coherence'] = 0
                features['max_coherence'] = 0
        else:
            features['mean_coherence'] = 0
            features['max_coherence'] = 0
        
        return features
    
    def extract_envelope_features(self, signal_data, channel_name):
        """Extract features from signal envelope"""
        # Hilbert transform for envelope
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        
        features = {}
        features[f'{channel_name}_envelope_mean'] = np.mean(envelope)
        features[f'{channel_name}_envelope_std'] = np.std(envelope)
        features[f'{channel_name}_envelope_max'] = np.max(envelope)
        features[f'{channel_name}_envelope_peak_factor'] = np.max(envelope) / (np.mean(envelope) + 1e-12)
        
        return features
    
    def extract_all_features(self, vibration1, vibration2):
        """Extract all features from both channels"""
        all_features = {}
        
        # Statistical features
        all_features.update(self.extract_statistical_features(vibration1, 'ch1'))
        all_features.update(self.extract_statistical_features(vibration2, 'ch2'))
        
        # Frequency features
        all_features.update(self.extract_frequency_features(vibration1, 'ch1'))
        all_features.update(self.extract_frequency_features(vibration2, 'ch2'))
        
        # Envelope features
        all_features.update(self.extract_envelope_features(vibration1, 'ch1'))
        all_features.update(self.extract_envelope_features(vibration2, 'ch2'))
        
        # Cross-channel features
        all_features.update(self.extract_cross_channel_features(vibration1, vibration2))
        
        return all_features

# ============================================
# PART 2: CLASSICAL ML CLASSIFIER
# ============================================

class ClassicalMLClassifier:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=42,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=10,
                random_state=42
            ),
            'Naive Bayes': GaussianNB()
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, labeled_segments, feature_extractor):
        """Convert labeled segments to feature matrix"""
        
        X_list = []
        y_list = []
        valid_indices = []
        
        print("\nExtracting features from segments...")
        for i, segment in enumerate(tqdm(labeled_segments, desc="Processing")):
            try:
                features = feature_extractor.extract_all_features(
                    segment['vibration1'],
                    segment['vibration2']
                )
                X_list.append(features)
                y_list.append(segment['label'])
                valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Error extracting features from segment {i}: {e}")
                continue
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        # Handle any NaN or infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        print(f"\n✅ Feature matrix shape: {X.shape}")
        print(f"✅ Number of classes: {len(y.unique())}")
        print(f"✅ Classes: {y.unique().tolist()}")
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        """Train multiple models and compare performance"""
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n📊 Dataset split:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            
            try:
                # Train
                model.fit(self.X_train_scaled, self.y_train)
                
                # Predict
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test, y_pred, average='weighted'
                )
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'true_values': self.y_test,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ Test Accuracy: {accuracy:.4f}")
                print(f"   ✅ Precision: {precision:.4f}")
                print(f"   ✅ Recall: {recall:.4f}")
                print(f"   ✅ F1-Score: {f1:.4f}")
                print(f"   ✅ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
        
        # Find best model based on F1 score
        if self.results:
            self.best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
            self.best_model = self.results[self.best_model_name]['model']
            
            # Get feature importance if available
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            print(f"\n{'🏆'*20}")
            print(f"🏆 BEST MODEL: {self.best_model_name}")
            print(f"   Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
            print(f"   F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
            print(f"{'🏆'*20}")
        
        return self.results
    
    def plot_results(self, X, y):
        """Create comprehensive visualization of results"""
        
        if not self.results:
            print("No results to plot")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model comparison bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in names]
            bars = ax1.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width*1.5)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2. Confusion matrix for best model
        ax2 = fig.add_subplot(gs[0, 2])
        best_results = self.results[self.best_model_name]
        cm = confusion_matrix(best_results['true_values'], best_results['predictions'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', ax=ax2, cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Normalized Count'})
        ax2.set_title(f'Confusion Matrix - {self.best_model_name}\n(Normalized)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        # 3. Classification report table
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')
        report = classification_report(best_results['true_values'],
                                      best_results['predictions'],
                                      target_names=self.label_encoder.classes_,
                                      output_dict=True)
        
        report_df = pd.DataFrame(report).transpose().round(3)
        
        # Create table
        cell_text = []
        for i, (idx, row) in enumerate(report_df.iterrows()):
            if i < len(report_df) - 3:  # Skip support rows
                cell_text.append([f"{row['precision']:.3f}", 
                                 f"{row['recall']:.3f}", 
                                 f"{row['f1-score']:.3f}",
                                 f"{int(row['support'])}"])
        
        table = ax3.table(cellText=cell_text,
                         colLabels=['Precision', 'Recall', 'F1', 'Support'],
                         rowLabels=self.label_encoder.classes_,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax3.set_title('Classification Report', fontsize=12, fontweight='bold')
        
        # 4. Feature importance
        if self.feature_importance is not None:
            ax4 = fig.add_subplot(gs[1, :2])
            top_features = self.feature_importance.head(20)
            bars = ax4.barh(range(len(top_features)), top_features['importance'].values)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['feature'].values, fontsize=9)
            ax4.set_xlabel('Importance')
            ax4.set_title(f'Top 20 Features - {self.best_model_name}', fontsize=12, fontweight='bold')
            ax4.invert_yaxis()
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
                ax4.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
        
        # 5. Cross-validation scores
        ax5 = fig.add_subplot(gs[1, 2])
        cv_means = [self.results[name]['cv_mean'] for name in names]
        cv_stds = [self.results[name]['cv_std'] for name in names]
        
        bars = ax5.bar(range(len(names)), cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax5.set_xlabel('Model')
        ax5.set_ylabel('CV Score')
        ax5.set_title('Cross-validation Scores', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(names)))
        ax5.set_xticklabels(names, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # Add value labels
        for i, (bar, mean) in enumerate(zip(bars, cv_means)):
            ax5.text(i, mean + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 6. ROC curves for best model
        if best_results['probabilities'] is not None:
            ax6 = fig.add_subplot(gs[1, 3])
            n_classes = len(self.label_encoder.classes_)
            
            # Binarize the labels for ROC
            y_test_bin = np.zeros((len(self.y_test), n_classes))
            for i, label in enumerate(self.y_test):
                y_test_bin[i, label] = 1
            
            for i, class_name in enumerate(self.label_encoder.classes_):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], best_results['probabilities'][:, i])
                roc_auc = auc(fpr, tpr)
                ax6.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', linewidth=2)
            
            ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax6.set_xlabel('False Positive Rate')
            ax6.set_ylabel('True Positive Rate')
            ax6.set_title('ROC Curves - One-vs-Rest', fontsize=12, fontweight='bold')
            ax6.legend(loc='lower right', fontsize=9)
            ax6.grid(True, alpha=0.3)
        
        # 7. Class distribution
        ax7 = fig.add_subplot(gs[2, 0])
        class_counts = pd.Series(y).value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        bars = ax7.bar(range(len(class_counts)), class_counts.values, color=colors)
        ax7.set_title('Class Distribution in Dataset', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Class')
        ax7.set_ylabel('Count')
        ax7.set_xticks(range(len(class_counts)))
        ax7.set_xticklabels(class_counts.index, rotation=45)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
            ax7.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # 8. Sample predictions (first 50)
        ax8 = fig.add_subplot(gs[2, 1:])
        y_true = best_results['true_values'][:50]
        y_pred = best_results['predictions'][:50]
        
        x_vals = np.arange(len(y_true))
        ax8.scatter(x_vals, y_true, c='blue', label='True', alpha=0.7, s=50, marker='o')
        ax8.scatter(x_vals, y_pred, c='red', label='Predicted', alpha=0.7, s=30, marker='x')
        ax8.set_xlabel('Sample Index')
        ax8.set_ylabel('Class')
        ax8.set_title('True vs Predicted (First 50 Test Samples)', fontsize=12, fontweight='bold')
        ax8.set_yticks(range(len(self.label_encoder.classes_)))
        ax8.set_yticklabels(self.label_encoder.classes_)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_classification_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n✅ Saved 'ml_classification_results.png'")
    
    def save_model(self, filename='best_ml_model.pkl'):
        """Save the best model and associated data"""
        import joblib
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance,
            'results': self.results,
            'best_model_name': self.best_model_name,
            'feature_names': self.X_train.columns.tolist() if hasattr(self, 'X_train') else None
        }
        
        joblib.dump(model_data, filename)
        print(f"\n✅ Saved best model ({self.best_model_name}) to '{filename}'")

# ============================================
# PART 3: DEEP LEARNING CLASSIFIER
# ============================================

class DeepLearningClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def build_cnn_model(self):
        """Build a 1D CNN for vibration classification"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv1D(64, kernel_size=20, activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv1D(128, kernel_size=15, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.2),
            
            # Third convolutional block
            layers.Conv1D(256, kernel_size=10, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.2),
            
            # Fourth convolutional block
            layers.Conv1D(512, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_model(self):
        """Build an LSTM model for vibration classification"""
        model = models.Sequential([
            # First LSTM layer
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            # Third LSTM layer
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_hybrid_model(self):
        """Build a hybrid CNN-LSTM model"""
        model = models.Sequential([
            # CNN layers for feature extraction
            layers.Conv1D(64, kernel_size=10, activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, kernel_size=10, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # LSTM for temporal dependencies
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, labeled_segments, max_length=5000):
        """Prepare raw signals for deep learning"""
        X_list = []
        y_list = []
        
        print("\nPreparing data for deep learning...")
        for segment in tqdm(labeled_segments, desc="Processing"):
            vib1 = segment['vibration1']
            vib2 = segment['vibration2']
            
            # Ensure consistent length (pad or truncate)
            if len(vib1) > max_length:
                vib1 = vib1[:max_length]
                vib2 = vib2[:max_length]
            elif len(vib1) < max_length:
                pad_len = max_length - len(vib1)
                vib1 = np.pad(vib1, (0, pad_len), 'constant', constant_values=0)
                vib2 = np.pad(vib2, (0, pad_len), 'constant', constant_values=0)
            
            # Stack as 2-channel input
            X_list.append(np.stack([vib1, vib2], axis=-1))
            y_list.append(segment['label'])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n✅ Deep Learning data shape: {X.shape}")
        print(f"✅ Labels shape: {y_encoded.shape}")
        print(f"✅ Classes: {self.label_encoder.classes_.tolist()}")
        
        return X, y_encoded
    
    def train(self, X, y, model_type='cnn', epochs=100, batch_size=32):
        """Train the deep learning model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize each channel independently
        print("\nNormalizing data...")
        for i in range(X_train.shape[-1]):
            # Calculate mean and std from training data
            mean = X_train[..., i].mean()
            std = X_train[..., i].std()
            
            # Normalize
            X_train[..., i] = (X_train[..., i] - mean) / (std + 1e-8)
            X_test[..., i] = (X_test[..., i] - mean) / (std + 1e-8)
        
        print(f"\n📊 Dataset split:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Build model
        if model_type == 'cnn':
            print(f"\nBuilding CNN model...")
            self.model = self.build_cnn_model()
        elif model_type == 'lstm':
            print(f"\nBuilding LSTM model...")
            self.model = self.build_lstm_model()
        else:
            print(f"\nBuilding Hybrid CNN-LSTM model...")
            self.model = self.build_hybrid_model()
        
        # Model summary
        self.model.summary()
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'best_dl_{model_type}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print(f"\nTraining {model_type.upper()} model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Test Accuracy: {test_acc:.4f}")
        
        return self.history, test_acc
    
    def plot_training_history(self, model_type='cnn'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title(f'{model_type.upper()} - Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title(f'{model_type.upper()} - Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'dl_{model_type}_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved 'dl_{model_type}_training_history.png'")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("="*70)
    print("GRADE 5: CLASSIFICATION OF RAILWAY INFRASTRUCTURE")
    print("="*70)
    
    # Step 1: Load labeled data from Grade 4
    print("\n📂 Step 1: Loading labeled segments from Grade 4...")
    try:
        with open('labeled_segments.pkl', 'rb') as f:
            labeled_segments = pickle.load(f)
        print(f"✅ Loaded {len(labeled_segments)} labeled segments")
        
        # Print label distribution
        label_counts = {}
        for seg in labeled_segments:
            label = seg['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count}")
            
    except FileNotFoundError:
        print("❌ Error: 'labeled_segments.pkl' not found.")
        print("   Please run Grade 4 first to generate labeled data.")
        return
    
    # ============================================
    # PART 1: CLASSICAL MACHINE LEARNING
    # ============================================
    
    print("\n" + "="*70)
    print("PART 1: CLASSICAL MACHINE LEARNING")
    print("="*70)
    
    # Extract features
    print("\n🔍 Step 2: Extracting features for classical ML...")
    feature_extractor = FeatureExtractor(sampling_rate=500)
    ml_classifier = ClassicalMLClassifier()
    
    X, y = ml_classifier.prepare_data(labeled_segments, feature_extractor)
    
    # Train and evaluate models
    print("\n🤖 Step 3: Training classical ML models...")
    results = ml_classifier.train_and_evaluate(X, y)
    
    # Plot results
    print("\n📊 Step 4: Visualizing ML results...")
    ml_classifier.plot_results(X, y)
    
    # Save best model
    ml_classifier.save_model('best_ml_model.pkl')
    
    # ============================================
    # PART 2: DEEP LEARNING
    # ============================================
    
    print("\n" + "="*70)
    print("PART 2: DEEP LEARNING")
    print("="*70)
    
    # Prepare data for deep learning
    print("\n🔧 Step 5: Preparing data for deep learning...")
    dl_classifier = DeepLearningClassifier(
        input_shape=(5000, 2),  # 10 seconds at 500Hz = 5000 samples, 2 channels
        num_classes=len(y.unique())
    )
    
    X_dl, y_dl = dl_classifier.prepare_data(labeled_segments)
    
    # Train CNN model
    print("\n🧠 Step 6: Training CNN model...")
    history_cnn, acc_cnn = dl_classifier.train(X_dl, y_dl, model_type='cnn', epochs=50)
    dl_classifier.plot_training_history('cnn')
    
    # Train LSTM model
    print("\n🧠 Step 7: Training LSTM model...")
    history_lstm, acc_lstm = dl_classifier.train(X_dl, y_dl, model_type='lstm', epochs=50)
    dl_classifier.plot_training_history('lstm')
    
    # Train Hybrid model
    print("\n🧠 Step 8: Training Hybrid CNN-LSTM model...")
    history_hybrid, acc_hybrid = dl_classifier.train(X_dl, y_dl, model_type='hybrid', epochs=50)
    dl_classifier.plot_training_history('hybrid')
    
    # ============================================
    # FINAL COMPARISON
    # ============================================
    
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON")
    print("="*70)
    
    # Collect all results
    all_results = []
    
    # Classical ML results
    print("\n📊 Classical ML Models:")
    for name, res in ml_classifier.results.items():
        print(f"   {name:20} - Accuracy: {res['accuracy']:.4f} | F1: {res['f1_score']:.4f}")
        all_results.append({
            'Model': f"ML: {name}",
            'Type': 'Classical ML',
            'Accuracy': res['accuracy'],
            'F1-Score': res['f1_score']
        })
    
    # Deep Learning results
    print("\n🧠 Deep Learning Models:")
    print(f"   CNN                 - Accuracy: {acc_cnn:.4f}")
    print(f"   LSTM                - Accuracy: {acc_lstm:.4f}")
    print(f"   Hybrid CNN-LSTM     - Accuracy: {acc_hybrid:.4f}")
    
    all_results.extend([
        {'Model': 'DL: CNN', 'Type': 'Deep Learning', 'Accuracy': acc_cnn, 'F1-Score': acc_cnn},
        {'Model': 'DL: LSTM', 'Type': 'Deep Learning', 'Accuracy': acc_lstm, 'F1-Score': acc_lstm},
        {'Model': 'DL: Hybrid', 'Type': 'Deep Learning', 'Accuracy': acc_hybrid, 'F1-Score': acc_hybrid}
    ])
    
    # Find overall best
    results_df = pd.DataFrame(all_results)
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    
    print("\n" + "🏆"*35)
    print(f"🏆 OVERALL BEST MODEL: {best_model['Model']}")
    print(f"🏆 Best Accuracy: {best_model['Accuracy']:.4f}")
    print(f"🏆 Model Type: {best_model['Type']}")
    print("🏆"*35)
    
    # Create final comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    ax = axes[0]
    models = results_df['Model'].tolist()
    accuracies = results_df['Accuracy'].tolist()
    colors = ['skyblue' if 'ML' in m else 'lightcoral' for m in models]
    bars = ax.barh(range(len(models)), accuracies, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.01, i, f'{acc:.4f}', va='center')
    
    # F1-Score comparison
    ax = axes[1]
    f1_scores = results_df['F1-Score'].tolist() if 'F1-Score' in results_df.columns else accuracies
    bars = ax.barh(range(len(models)), f1_scores, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('F1-Score')
    ax.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax.text(f1 + 0.01, i, f'{f1:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('final_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Saved 'final_model_comparison.png'")
    
    # Save comparison results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("✅ Saved 'model_comparison_results.csv'")
    
    print("\n" + "="*70)
    print("✅ GRADE 5 COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()