"""
Save and load trained classifier for production inference.
"""
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from own_signal_processing import WristGaitDetector, load_wisdm_6axis, resample_to_100hz
import os

def save_trained_model(model, scaler, feature_names, filepath='gait_classifier.pkl'):
    """Save trained classifier and scaler to disk."""
    package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    with open(filepath, 'wb') as f:
        pickle.dump(package, f)
    print(f"Model saved to {filepath}")

def load_trained_model(filepath='gait_classifier.pkl'):
    """Load trained classifier and scaler from disk."""
    with open(filepath, 'rb') as f:
        package = pickle.load(f)
    return package['model'], package['scaler'], package['feature_names']

def predict_gait_with_classifier(data_df, classifier, scaler, detector_for_features):
    """
    Predict gait labels for a resampled 100Hz DataFrame using trained classifier.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Resampled IMU data with columns: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, activity, timestamp
    classifier : RandomForestClassifier
        Trained classifier
    scaler : StandardScaler
        Fitted scaler for normalization
    detector_for_features : WristGaitDetector
        Detector instance to extract features from windows
    
    Returns:
    --------
    predictions : np.array (0/1)
        Binary predictions per window
    timestamps : np.array
        Start time of each window
    confidences : np.array
        Confidence scores (probability of gait class)
    """
    preds, ts, confs, features_list = detector_for_features.detect(data_df, return_features=True)
    
    # Build feature vectors from extracted features
    X = []
    for features in features_list:
        fvec = [
            features['acc_amplitude'],
            features['acc_range'],
            features['gyr_amplitude'],
            features['dominant_freq'],
            features['spectral_power'],
            features['harmonic_ratio'],
            features['gyr_dominant_freq'],
            features['gyr_spectral_power'],
            features['gyr_harmonic_ratio'],
            features['max_autocorr'],
            features['regularity'],
            features['jerk'],
            features['acc_energy']
        ]
        X.append(fvec)
    
    X = np.array(X)
    X_scaled = scaler.transform(X)
    
    # Predict using classifier
    y_pred = classifier.predict(X_scaled)
    y_pred_proba = classifier.predict_proba(X_scaled)[:, 1]  # prob of gait class
    
    return y_pred, ts, y_pred_proba

if __name__ == '__main__':
    # Example: Load a test file and make predictions
    import glob
    
    BASE = os.path.join('wisdm-dataset', 'raw', 'watch', 'accel')
    test_file = sorted(glob.glob(os.path.join(BASE, '*.txt')))[47]
    
    # Load and prepare data
    print(f"Loading test file: {test_file}")
    df_raw = load_wisdm_6axis(test_file)
    df100 = resample_to_100hz(df_raw)
    df100 = df100.iloc[:int(120*100)].reset_index(drop=True)  # 120s for speed
    
    # Check if saved model exists
    if os.path.exists('gait_classifier.pkl'):
        print("Loading trained classifier...")
        clf, scl, feat_names = load_trained_model('gait_classifier.pkl')
        detector = WristGaitDetector()
        
        # Make predictions
        y_pred, ts, y_conf = predict_gait_with_classifier(df100, clf, scl, detector)
        
        print(f"\nPredictions on {len(y_pred)} windows:")
        print(f"  Gait: {np.sum(y_pred)} ({100*np.sum(y_pred)/len(y_pred):.1f}%)")
        print(f"  Non-gait: {len(y_pred)-np.sum(y_pred)} ({100*(1-np.sum(y_pred)/len(y_pred)):.1f}%)")
        print(f"  Avg confidence: {np.mean(y_conf):.3f}")
    else:
        print("Saved model not found. Run train_classifier.py first to train and save the model.")
