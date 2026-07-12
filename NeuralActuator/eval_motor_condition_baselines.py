#!/usr/bin/env python3
"""
Baseline methods for motor condition detection.

1. Current Threshold: Simple thresholding on mean |current|
2. SVM with handcrafted features: Uses current, velocity, tracking error features
3. Current-Velocity Ratio: Physics-inspired, normalizes current by motion

Rationale: A degraded motor (e.g., with increased mechanical resistance)
requires higher current to achieve the same position trajectory.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from scipy.spatial.distance import cdist
import json


def load_csv_data(csv_path):
    """Load CSV and return relevant columns."""
    df = pd.read_csv(csv_path)

    # All relevant columns
    pos_cols = ['pos1', 'pos2', 'pos3', 'pos4', 'pos5']
    goal_cols = ['goal_pos1', 'goal_pos2', 'goal_pos3', 'goal_pos4', 'goal_pos5']
    vel_cols = ['vel1', 'vel2', 'vel3', 'vel4', 'vel5']
    current_cols = ['current1', 'current2', 'current3', 'current4', 'current5']

    return {
        'positions': df[pos_cols].values,
        'goals': df[goal_cols].values,
        'velocities': df[vel_cols].values,
        'currents': df[current_cols].values,
        'n_samples': len(df)
    }


def extract_window_features(data, joint=3, window_size=64):
    """Extract rich features for sliding windows including temporal and frequency features."""
    j = joint - 1  # 0-indexed
    n_samples = data['n_samples']

    pos = data['positions'][:, j]
    goal = data['goals'][:, j]
    vel = data['velocities'][:, j]
    curr = data['currents'][:, j]

    # Extract features for each window
    window_features = []
    window_raw_sequences = []  # Store raw sequences for temporal models

    for start in range(0, n_samples - window_size, window_size // 2):  # 50% overlap
        end = start + window_size

        w_curr = curr[start:end]
        w_vel = vel[start:end]
        w_pos = pos[start:end]
        w_goal = goal[start:end]

        # === Basic Statistics ===
        mean_current = np.mean(np.abs(w_curr))
        std_current = np.std(w_curr)
        max_current = np.max(np.abs(w_curr))
        min_current = np.min(np.abs(w_curr))

        # Higher-order statistics
        from scipy.stats import skew, kurtosis
        skew_current = skew(w_curr)
        kurt_current = kurtosis(w_curr)

        # Tracking error
        tracking_error = np.abs(w_goal - w_pos)
        mean_tracking_error = np.mean(tracking_error)
        max_tracking_error = np.max(tracking_error)
        std_tracking_error = np.std(tracking_error)

        # === Temporal Features ===
        # Current derivative (rate of change)
        curr_diff = np.diff(w_curr)
        mean_curr_derivative = np.mean(np.abs(curr_diff))
        max_curr_derivative = np.max(np.abs(curr_diff))

        # Velocity derivative (acceleration)
        vel_diff = np.diff(w_vel)
        mean_acceleration = np.mean(np.abs(vel_diff))

        # Trend (linear regression slope)
        t = np.arange(len(w_curr))
        curr_trend = np.polyfit(t, w_curr, 1)[0]  # slope

        # === Frequency Features (FFT) ===
        fft_curr = np.fft.fft(w_curr)
        fft_magnitude = np.abs(fft_curr[:len(fft_curr)//2])

        # Spectral features
        total_power = np.sum(fft_magnitude**2)
        dominant_freq_idx = np.argmax(fft_magnitude[1:]) + 1  # skip DC
        dominant_freq_power = fft_magnitude[dominant_freq_idx]**2 / (total_power + 1e-10)

        # Low/high frequency power ratio
        mid_idx = len(fft_magnitude) // 4
        low_freq_power = np.sum(fft_magnitude[:mid_idx]**2)
        high_freq_power = np.sum(fft_magnitude[mid_idx:]**2)
        freq_ratio = low_freq_power / (high_freq_power + 1e-10)

        # Spectral entropy
        fft_prob = fft_magnitude**2 / (total_power + 1e-10)
        fft_prob = fft_prob[fft_prob > 0]
        spectral_entropy = -np.sum(fft_prob * np.log(fft_prob + 1e-10))

        # === Physics-inspired Features ===
        motion_mask = np.abs(w_vel) > 0.05
        if motion_mask.sum() > 10:
            current_during_motion = np.mean(np.abs(w_curr[motion_mask]))
            vel_during_motion = np.mean(np.abs(w_vel[motion_mask]))
            current_vel_ratio = current_during_motion / (vel_during_motion + 0.01)

            # Power consumption estimate (current * velocity ~ torque * angular velocity)
            power_estimate = np.mean(np.abs(w_curr[motion_mask] * w_vel[motion_mask]))
        else:
            current_during_motion = mean_current
            current_vel_ratio = mean_current
            power_estimate = 0.0

        # === Multi-scale Features ===
        # Split window into halves and compare
        half = window_size // 2
        mean_current_first_half = np.mean(np.abs(w_curr[:half]))
        mean_current_second_half = np.mean(np.abs(w_curr[half:]))
        current_asymmetry = mean_current_second_half - mean_current_first_half

        window_features.append({
            # Basic
            'mean_current': mean_current,
            'std_current': std_current,
            'max_current': max_current,
            'min_current': min_current,
            'skew_current': skew_current,
            'kurt_current': kurt_current,
            # Tracking
            'mean_tracking_error': mean_tracking_error,
            'max_tracking_error': max_tracking_error,
            'std_tracking_error': std_tracking_error,
            # Temporal
            'mean_curr_derivative': mean_curr_derivative,
            'max_curr_derivative': max_curr_derivative,
            'mean_acceleration': mean_acceleration,
            'curr_trend': curr_trend,
            # Frequency
            'dominant_freq_power': dominant_freq_power,
            'freq_ratio': freq_ratio,
            'spectral_entropy': spectral_entropy,
            # Physics
            'current_during_motion': current_during_motion,
            'current_vel_ratio': current_vel_ratio,
            'power_estimate': power_estimate,
            # Multi-scale
            'current_asymmetry': current_asymmetry,
        })

        # Store raw sequence for temporal models
        window_raw_sequences.append({
            'current': w_curr.copy(),
            'velocity': w_vel.copy(),
            'position': w_pos.copy(),
            'goal': w_goal.copy(),
        })

    return window_features, window_raw_sequences


def fast_dtw_distance(seq1, seq2, radius=5):
    """Fast DTW with constraint window (O(n) instead of O(n^2))."""
    n, m = len(seq1), len(seq2)

    # Subsample for speed
    step = max(1, min(n, m) // 16)
    s1 = seq1[::step]
    s2 = seq2[::step]
    n, m = len(s1), len(s2)

    # DTW with Sakoe-Chiba band constraint
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius) + 1
        for j in range(j_start, j_end):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix[n, m]


def compute_dtw_kernel(X_raw, Y_raw=None, max_samples=200):
    """Compute DTW-based kernel matrix for time series SVM.

    Uses subsampling for large datasets to keep computation tractable.
    """
    if Y_raw is None:
        Y_raw = X_raw
        symmetric = True
    else:
        symmetric = False

    n = len(X_raw)
    m = len(Y_raw)
    dist_matrix = np.zeros((n, m))

    total = n * m if not symmetric else n * (n + 1) // 2
    print(f"    Computing {total} DTW distances...", end=" ", flush=True)

    count = 0
    for i in range(n):
        j_start = i if symmetric else 0
        for j in range(j_start, m):
            dist_matrix[i, j] = fast_dtw_distance(X_raw[i]['current'], Y_raw[j]['current'])
            if symmetric and i != j:
                dist_matrix[j, i] = dist_matrix[i, j]
            count += 1
            if count % 5000 == 0:
                print(f"{count}/{total}", end=" ", flush=True)

    print("Done!")

    # Convert distance to similarity (RBF-like kernel)
    gamma = 1.0 / (np.median(dist_matrix) + 1e-10)
    kernel_matrix = np.exp(-gamma * dist_matrix)
    return kernel_matrix


def compute_mean_current(data, joint=3):
    """Compute mean absolute current for specified joint (1-indexed)."""
    currents = data['currents']
    mean_current = np.mean(np.abs(currents[:, joint-1]))
    return mean_current


def evaluate_at_threshold(all_currents, all_labels, threshold):
    """Evaluate metrics at a specific threshold."""
    preds = [1 if c > threshold else 0 for c in all_currents]
    labels = np.array(all_labels)
    preds = np.array(preds)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'preds': preds}


def find_optimal_threshold(all_currents, all_labels, n_thresholds=100):
    """Find optimal threshold using grid search."""
    min_val, max_val = min(all_currents), max(all_currents)

    best_acc = 0
    best_threshold = min_val
    best_preds = None

    for threshold in np.linspace(min_val, max_val, n_thresholds):
        preds = [1 if c > threshold else 0 for c in all_currents]
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_preds = preds

    return best_threshold, best_acc, best_preds


def main():
    parser = argparse.ArgumentParser(description='Evaluate motor condition detection baseline')
    parser.add_argument('--output', type=str, default='outputs/motor_condition_baseline.json',
                        help='Output JSON file')
    parser.add_argument('--window_size', type=int, default=64, help='Window size for feature extraction')
    parser.add_argument('--ours_json', type=str, default=None,
                        help='evaluate_actuator output JSON; reads CLASSIFICATION_J3 metrics for the Ours column')
    args = parser.parse_args()

    # Optional: load our model's metrics from an evaluate_actuator output JSON
    ours = None
    if args.ours_json is not None:
        with open(args.ours_json) as f:
            eval_results = json.load(f)
        cls_j3 = eval_results['CLASSIFICATION_J3']
        ours = {
            'accuracy': float(cls_j3['accuracy']),
            'precision': float(cls_j3['precision']),
            'recall': float(cls_j3['recall']),
            'auc_roc': float(cls_j3['auc_roc']),
        }

    # Training datasets (same as benchmark_motor_condition.yaml)
    # Use train data to train baselines, test data to evaluate (fair comparison with neural network)
    train_datasets = [
        # Normal motor
        ('normal', 'data/motor_condition/pick_place_empty/train/001.csv'),
        ('normal', 'data/motor_condition/pick_place_empty/train/002.csv'),
        ('normal', 'data/motor_condition/pick_place_empty/train/003.csv'),
        ('normal', 'data/motor_condition/pick_place_empty/train/004.csv'),
        ('normal', 'data/motor_condition/pick_place_object_200g/train/001.csv'),
        ('normal', 'data/motor_condition/pick_place_object_200g/train/002.csv'),
        ('normal', 'data/motor_condition/pick_place_object_200g/train/003.csv'),
        ('normal', 'data/motor_condition/pick_place_object_200g/train/004.csv'),
        # Degraded motor
        ('degraded', 'data/motor_condition/pick_place_empty_degrade/train/001.csv'),
        ('degraded', 'data/motor_condition/pick_place_empty_degrade/train/002.csv'),
        ('degraded', 'data/motor_condition/pick_place_empty_degrade/train/003.csv'),
        ('degraded', 'data/motor_condition/pick_place_empty_degrade/train/004.csv'),
        ('degraded', 'data/motor_condition/pick_place_object_200g_degrade/train/001.csv'),
        ('degraded', 'data/motor_condition/pick_place_object_200g_degrade/train/002.csv'),
        ('degraded', 'data/motor_condition/pick_place_object_200g_degrade/train/003.csv'),
        ('degraded', 'data/motor_condition/pick_place_object_200g_degrade/train/004.csv'),
    ]

    test_datasets = [
        ('normal', 'data/motor_condition/pick_place_empty/test/001.csv'),
        ('normal', 'data/motor_condition/pick_place_object_200g/test/001.csv'),
        ('degraded', 'data/motor_condition/pick_place_empty_degrade/test/001.csv'),
        ('degraded', 'data/motor_condition/pick_place_object_200g_degrade/test/001.csv'),
    ]

    print("=" * 60)
    print("Motor Condition Detection - Baseline Evaluation")
    print(f"Window-level evaluation (window_size={args.window_size})")
    print("=" * 60)

    # Load training data
    print("\n[Training Data]")
    train_features = []
    train_raw = []
    train_labels = []

    for label_str, csv_path in train_datasets:
        if not Path(csv_path).exists():
            continue
        data = load_csv_data(csv_path)
        features, raw_seqs = extract_window_features(data, joint=3, window_size=args.window_size)
        label = 1 if label_str == 'degraded' else 0

        for f, r in zip(features, raw_seqs):
            train_features.append(f)
            train_raw.append(r)
            train_labels.append(label)

    print(f"  Total: {len(train_features)} windows")
    print(f"  Normal: {train_labels.count(0)}, Degraded: {train_labels.count(1)}")

    # Load test data
    print("\n[Test Data]")
    test_features = []
    test_raw = []
    test_labels = []
    trajectory_info = []

    for label_str, csv_path in test_datasets:
        if not Path(csv_path).exists():
            print(f"  WARNING: {csv_path} not found, skipping...")
            continue

        data = load_csv_data(csv_path)
        features, raw_seqs = extract_window_features(data, joint=3, window_size=args.window_size)
        label = 1 if label_str == 'degraded' else 0

        for f, r in zip(features, raw_seqs):
            test_features.append(f)
            test_raw.append(r)
            test_labels.append(label)

        task_name = Path(csv_path).parent.parent.name + "/" + Path(csv_path).stem
        trajectory_info.append({
            'name': task_name,
            'n_windows': len(features),
            'label': label_str.upper(),
            'mean_current': np.mean([w['mean_current'] for w in features])
        })

        print(f"  {task_name}: {len(features)} windows ({label_str.upper()})")

    n_test = len(test_features)
    n_test_normal = sum(1 for l in test_labels if l == 0)
    n_test_degraded = sum(1 for l in test_labels if l == 1)
    print(f"  Total: {n_test} windows (Normal: {n_test_normal}, Degraded: {n_test_degraded})")

    # Convert to arrays
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # All feature names (21 features total)
    feature_names = [
        # Basic (6)
        'mean_current', 'std_current', 'max_current', 'min_current', 'skew_current', 'kurt_current',
        # Tracking (3)
        'mean_tracking_error', 'max_tracking_error', 'std_tracking_error',
        # Temporal (4)
        'mean_curr_derivative', 'max_curr_derivative', 'mean_acceleration', 'curr_trend',
        # Frequency (3)
        'dominant_freq_power', 'freq_ratio', 'spectral_entropy',
        # Physics (3)
        'current_during_motion', 'current_vel_ratio', 'power_estimate',
        # Multi-scale (1)
        'current_asymmetry',
    ]

    # Create feature matrices
    X_train = np.array([[w[name] for name in feature_names] for w in train_features])
    X_test = np.array([[w[name] for name in feature_names] for w in test_features])

    print(f"\nFeatures ({len(feature_names)}): Basic(6), Tracking(3), Temporal(4), Frequency(3), Physics(3), Multi-scale(1)")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Normalize features (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================================================================
    # Baseline 1: Current Threshold (learned on train, evaluated on test)
    # ================================================================
    print("\n" + "-" * 60)
    print("Baseline 1: Mean Current Threshold")
    print("-" * 60)

    # Find optimal threshold on training data
    train_currents = [w['mean_current'] for w in train_features]
    threshold, _, _ = find_optimal_threshold(train_currents, train_labels)

    # Evaluate on test data
    test_currents = [w['mean_current'] for w in test_features]
    preds = [1 if c > threshold else 0 for c in test_currents]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, test_currents)
    except:
        auc = 0.5

    print(f"  Threshold (from train): {threshold:.1f} mA")
    print(f"  Test Accuracy:  {acc*100:.1f}%")
    print(f"  Test Precision: {prec*100:.1f}%")
    print(f"  Test Recall:    {rec*100:.1f}%")
    print(f"  Test AUC-ROC:   {auc:.3f}")

    # ================================================================
    # Baseline 2: SVM with rich handcrafted features
    # ================================================================
    print("\n" + "-" * 60)
    print("Baseline 2: SVM with Rich Features (21D)")
    print("-" * 60)

    # SVM with RBF kernel - train on train data
    svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    svm.fit(X_train_scaled, y_train)

    # Evaluate on test data
    svm_preds = svm.predict(X_test_scaled)
    svm_probs = svm.predict_proba(X_test_scaled)[:, 1]

    svm_acc = accuracy_score(y_test, svm_preds)
    svm_prec = precision_score(y_test, svm_preds, zero_division=0)
    svm_rec = recall_score(y_test, svm_preds, zero_division=0)
    try:
        svm_auc = roc_auc_score(y_test, svm_probs)
    except:
        svm_auc = 0.5

    print(f"  Test Accuracy:  {svm_acc*100:.1f}%")
    print(f"  Test Precision: {svm_prec*100:.1f}%")
    print(f"  Test Recall:    {svm_rec*100:.1f}%")
    print(f"  Test AUC-ROC:   {svm_auc:.3f}")

    # ================================================================
    # Baseline 3: Random Forest with rich handcrafted features
    # ================================================================
    print("\n" + "-" * 60)
    print("Baseline 3: Random Forest with Rich Features (21D)")
    print("-" * 60)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    rf_preds = rf.predict(X_test_scaled)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds, zero_division=0)
    rf_rec = recall_score(y_test, rf_preds, zero_division=0)
    try:
        rf_auc = roc_auc_score(y_test, rf_probs)
    except:
        rf_auc = 0.5

    print(f"  Test Accuracy:  {rf_acc*100:.1f}%")
    print(f"  Test Precision: {rf_prec*100:.1f}%")
    print(f"  Test Recall:    {rf_rec*100:.1f}%")
    print(f"  Test AUC-ROC:   {rf_auc:.3f}")

    # Top 5 feature importance
    importance_dict = dict(zip(feature_names, rf.feature_importances_))
    top5 = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 features: {[(k, round(v, 3)) for k, v in top5]}")

    # ================================================================
    # Summary: Find best baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary of All Baselines")
    print("=" * 60)

    baselines = {
        'Threshold': {'accuracy': acc, 'precision': prec, 'recall': rec, 'auc_roc': auc},
        'SVM': {'accuracy': svm_acc, 'precision': svm_prec, 'recall': svm_rec, 'auc_roc': svm_auc},
        'RF': {'accuracy': rf_acc, 'precision': rf_prec, 'recall': rf_rec, 'auc_roc': rf_auc},
    }

    for name, metrics in baselines.items():
        print(f"  {name}: Acc={metrics['accuracy']*100:.1f}%, Prec={metrics['precision']*100:.1f}%, Rec={metrics['recall']*100:.1f}%, AUC={metrics['auc_roc']:.3f}")

    # Best baseline by accuracy
    best_baseline_name = max(baselines.keys(), key=lambda k: baselines[k]['accuracy'])
    best_baseline = baselines[best_baseline_name]
    print(f"\n  Best Baseline: {best_baseline_name}")

    # Print LaTeX tables
    print("\n" + "=" * 60)
    print("LaTeX Tables")
    print("=" * 60)

    # Appendix table: All baselines comparison
    print("\n% === APPENDIX TABLE: Baseline Methods Comparison ===")
    print(r"""\begin{table}[h]
  \centering
  \caption{\textbf{Comparison of baseline methods for motor condition detection.} We evaluate window-level classification using: (1) current magnitude thresholding, (2) SVM with handcrafted features, and (3) Random Forest with handcrafted features. Features include mean current, current std, current-velocity ratio, tracking error, and current during motion. All methods are evaluated on the same test set.}
  \label{tab:motor-condition-baselines}
  \begin{tabular}{lcccc}
    \toprule
    \textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{AUC-ROC} \\
    \midrule""")

    for name, metrics in baselines.items():
        print(f"    {name} & {metrics['accuracy']*100:.1f}\\% & {metrics['precision']*100:.1f}\\% & {metrics['recall']*100:.1f}\\% & {metrics['auc_roc']:.3f} \\\\")

    if ours is not None:
        print(r"    \midrule")
        print(f"    \\textbf{{Ours}} & \\textbf{{{ours['accuracy']*100:.1f}\\%}} & \\textbf{{{ours['precision']*100:.1f}\\%}} & \\textbf{{{ours['recall']*100:.1f}\\%}} & \\textbf{{{ours['auc_roc']:.3f}}} \\\\")
    print(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

    # Main table: all baselines (plus ours when --ours_json is given)
    print("\n% === MAIN TABLE: All Baselines vs Ours ===")
    th = baselines['Threshold']
    sv = baselines['SVM']
    rf = baselines['RF']
    print(r"""\begin{wraptable}{r}{0.45\columnwidth}
\vspace{-4.5mm}
  \centering
  \caption{\textbf{Motor condition detection.}}
  \label{tab:motor-condition-detection}
  \vspace{-1.5mm}
\hspace{-4mm}
  \resizebox{0.47\columnwidth}{!}{""")
    if ours is not None:
        print(r"""  \begin{tabular}{lcccc}
    \toprule
    \textbf{Metric} & Thres. & SVM & RF & \textbf{Ours} \\
    \midrule""")
        print(f"    Accuracy  & {th['accuracy']*100:.1f}\\% & {sv['accuracy']*100:.1f}\\% & {rf['accuracy']*100:.1f}\\% & \\textbf{{{ours['accuracy']*100:.1f}\\%}} \\\\")
        print(f"    Precision & {th['precision']*100:.1f}\\% & {sv['precision']*100:.1f}\\% & {rf['precision']*100:.1f}\\% & \\textbf{{{ours['precision']*100:.1f}\\%}} \\\\")
        print(f"    Recall    & {th['recall']*100:.1f}\\% & {sv['recall']*100:.1f}\\% & {rf['recall']*100:.1f}\\% & \\textbf{{{ours['recall']*100:.1f}\\%}} \\\\")
        print(f"    AUC-ROC   & {th['auc_roc']:.2f} & {sv['auc_roc']:.2f} & {rf['auc_roc']:.2f} & \\textbf{{{ours['auc_roc']:.2f}}} \\\\")
    else:
        print(r"""  \begin{tabular}{lccc}
    \toprule
    \textbf{Metric} & Thres. & SVM & RF \\
    \midrule""")
        print(f"    Accuracy  & {th['accuracy']*100:.1f}\\% & {sv['accuracy']*100:.1f}\\% & {rf['accuracy']*100:.1f}\\% \\\\")
        print(f"    Precision & {th['precision']*100:.1f}\\% & {sv['precision']*100:.1f}\\% & {rf['precision']*100:.1f}\\% \\\\")
        print(f"    Recall    & {th['recall']*100:.1f}\\% & {sv['recall']*100:.1f}\\% & {rf['recall']*100:.1f}\\% \\\\")
        print(f"    AUC-ROC   & {th['auc_roc']:.2f} & {sv['auc_roc']:.2f} & {rf['auc_roc']:.2f} \\\\")
    print(r"""    \bottomrule
  \end{tabular}
  }
\vspace{-4mm}
\end{wraptable}
""")

    # Save results
    results = {
        'train_windows': len(train_features),
        'test_windows': n_test,
        'test_normal': n_test_normal,
        'test_degraded': n_test_degraded,
        'n_features': len(feature_names),
        'baselines': {
            'threshold': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'auc_roc': float(auc),
                'threshold': float(threshold)
            },
            'svm': {
                'accuracy': float(svm_acc),
                'precision': float(svm_prec),
                'recall': float(svm_rec),
                'auc_roc': float(svm_auc)
            },
            'random_forest': {
                'accuracy': float(rf_acc),
                'precision': float(rf_prec),
                'recall': float(rf_rec),
                'auc_roc': float(rf_auc)
            }
        },
        'best_baseline': best_baseline_name,
    }
    if ours is not None:
        results['ours'] = ours

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
