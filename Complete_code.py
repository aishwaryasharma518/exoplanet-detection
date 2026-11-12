# final_complete_solution.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import joblib
import os

print("=" * 60)
print("FINAL COMPLETE SOLUTION")
print("=" * 60)

# ============================================
# STEP 1: CREATE DATASET
# ============================================
print("\n📊 Creating dataset...")

all_data = []
all_labels = []

# Time array
time = np.linspace(0, 100, 20000)

for star_id in range(20):
    is_planet = star_id < 10
    
    # Start with flat flux
    flux = np.ones_like(time)
    
    # Add noise
    np.random.seed(star_id)
    noise = np.random.normal(0, 0.0003, len(flux))
    flux = flux + noise
    
    if is_planet:
        # Add transits for planet stars
        transit_times = [25, 50, 75]
        for tt in transit_times:
            mask = np.abs(time - tt) < 0.08
            flux[mask] = flux[mask] * 0.9815  # 1.85% depth
        star_type = "PLANET"
    else:
        star_type = "NO PLANET"
    
    # Normalize
    flux = flux / np.mean(flux)
    
    # Save to CSV
    filename = f'star_{star_id}_planet.csv' if is_planet else f'star_{star_id}_noplanet.csv'
    df = pd.DataFrame({'time': time, 'flux': flux})
    df.to_csv(filename, index=False)
    
    # Calculate transit depth for verification
    depth = (1 - flux.min()) * 100
    print(f"   {filename}: Depth = {depth:.4f}% ({star_type})")
    
    # Extract features immediately
    threshold = 0.995
    dip_points = flux[flux < threshold]
    window = min(100, len(flux) // 10)
    if window < 2:
        window = 2
    smoothed = np.convolve(flux, np.ones(window)/window, mode='same')
    noise_std = np.std(flux - smoothed)
    
    features = [
        (1 - flux.min()) * 100,                           # transit_depth_percent
        len(dip_points),                                   # num_dips
        (1 - flux.min()) / noise_std if noise_std > 0 else 0,  # signal_to_noise
        1 - flux.min(),                                    # deepest_dip
        len(dip_points) / len(flux) * 100,                 # dip_ratio
        len(dip_points),                                   # transit_count
        flux.min(),                                        # min_flux
        np.std(flux),                                      # std_flux
        stats.skew(flux),                                  # skewness
        stats.kurtosis(flux),                              # kurtosis
        np.percentile(flux, 75) - np.percentile(flux, 25)  # iqr
    ]
    
    all_data.append(features)
    all_labels.append(1 if is_planet else 0)

# ============================================
# STEP 2: CREATE FEATURE MATRIX
# ============================================
print("\n📊 Creating feature matrix...")

X = np.array(all_data)
y = np.array(all_labels)

feature_names = [
    'transit_depth_percent', 'num_dips', 'signal_to_noise', 'deepest_dip',
    'dip_ratio', 'transit_count', 'min_flux', 'std_flux', 'skewness', 'kurtosis', 'iqr'
]

feature_df = pd.DataFrame(X, columns=feature_names)
feature_df['star_label'] = y
feature_df.to_csv('feature_matrix.csv', index=False)
print(f"   ✅ Saved feature_matrix.csv ({len(feature_df)} rows, {len(feature_names)} features)")

# ============================================
# STEP 3: TRAIN MODEL
# ============================================
print("\n🌲 Training Random Forest...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Test predictions
y_pred = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(f"   Test Accuracy: {accuracy*100:.1f}%")

# Save model
joblib.dump(rf_model, 'random_forest_model.pkl')
print("   ✅ Saved random_forest_model.pkl")

# ============================================
# STEP 4: TEST EACH STAR
# ============================================
print("\n" + "=" * 60)
print("📊 FINAL TEST - PREDICTIONS")
print("=" * 60)

for i in range(20):
    filename = f'star_{i}_planet.csv' if i < 10 else f'star_{i}_noplanet.csv'
    df = pd.read_csv(filename)
    flux = df['flux'].values
    
    # Extract features for this star
    threshold = 0.995
    dip_points = flux[flux < threshold]
    window = min(100, len(flux) // 10)
    if window < 2:
        window = 2
    smoothed = np.convolve(flux, np.ones(window)/window, mode='same')
    noise_std = np.std(flux - smoothed)
    
    features = np.array([[
        (1 - flux.min()) * 100,
        len(dip_points),
        (1 - flux.min()) / noise_std if noise_std > 0 else 0,
        1 - flux.min(),
        len(dip_points) / len(flux) * 100,
        len(dip_points),
        flux.min(),
        np.std(flux),
        stats.skew(flux),
        stats.kurtosis(flux),
        np.percentile(flux, 75) - np.percentile(flux, 25)
    ]])
    
    pred = rf_model.predict(features)[0]
    actual = "PLANET" if i < 10 else "NO PLANET"
    result = "✅" if (pred == 1 and i < 10) or (pred == 0 and i >= 10) else "❌"
    pred_text = "PLANET" if pred == 1 else "NO PLANET"
    depth = (1 - flux.min()) * 100
    
    print(f"   {result} {filename}: Depth={depth:.3f}% | Actual={actual} | Predicted={pred_text}")

print("\n" + "=" * 60)
print("✅ COMPLETE!")
print("=" * 60)
print("\n👉 Now run the UI:")
print("   streamlit run 09_exoplanet_ui.py")