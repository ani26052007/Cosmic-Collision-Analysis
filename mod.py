import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import miceforest as mf

import sklearn

from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')

X = data.drop(columns=['hazardous'])

Y = data['hazardous']

X_train, X_test, y_train, y_test = train_test_split(

X, 

Y, 

test_size=0.2, 

random_state=42

)

print(X.shape,Y.shape)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)


print("Let us look at percentage of missing values per column")
# 1. Identify and Quantify
missing_counts = X_train.isnull().sum()
missing_percentage = (X_train.isnull().mean() * 100)

# Combine into a summary table for easy viewing
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage (%)': missing_percentage,
    'dtype': X_train.dtypes
}).sort_values(by='Percentage (%)', ascending=False)

print(missing_summary)
columns_to_drop_ = ['miss_dist_miles', 'miss_dist_kilometers', 'miss_dist_lunar', 'relative_velocity_km_per_hr' ]
X_train =  X_train.drop(columns = columns_to_drop_)
X_test =  X_test.drop(columns = columns_to_drop_)

missing_counts_1 = X_train.isnull().sum()
missing_percentage_1 = (X_train.isnull().mean() * 100)

# Combine into a summary table for easy viewing
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts_1,
    'Percentage (%)': missing_percentage_1,
    'dtype': X_train.dtypes
}).sort_values(by='Percentage (%)', ascending=False)

print(missing_summary)
plt.figure(figsize=(20,10))
sns.heatmap(X_train.isnull(), cbar=False, cmap='viridis')
plt.title("Pattern of Missing Values")
plt.show()
def physics_based_imputation_v2(df):
    """
    Fill ONLY the physics-constrained relationships
    Leave the rest for MICE
    """
    df = df.copy()
    
    # Constants
    AU = 1.496e11
    G = 6.67430e-11
    M_SUN = 1.989e30
    
    # 1. Semi-major axis ↔ Mean motion (Kepler's 3rd Law)
    mask = df['mean_motion'].isna() & df['semi_major_axis'].notna()
    if mask.sum() > 0:
        a_m = df.loc[mask, 'semi_major_axis'] * AU
        n_rad_per_sec = np.sqrt(G * M_SUN / a_m**3)
        df.loc[mask, 'mean_motion'] = n_rad_per_sec * (180/np.pi) * 86400
        print(f"✓ Physics: Filled {mask.sum()} mean_motion from semi_major_axis")
    
    mask = df['semi_major_axis'].isna() & df['mean_motion'].notna()
    if mask.sum() > 0:
        n_rad_per_sec = df.loc[mask, 'mean_motion'] * (np.pi/180) / 86400
        a_m = (G * M_SUN / n_rad_per_sec**2)**(1/3)
        df.loc[mask, 'semi_major_axis'] = a_m / AU
        print(f"✓ Physics: Filled {mask.sum()} semi_major_axis from mean_motion")
    
    # 2. Calculate eccentricity where possible
    mask = df['aphelion_dist'].notna() & df['semi_major_axis'].notna()
    if mask.sum() > 0:
        if 'eccentricity' not in df.columns:
            df['eccentricity'] = np.nan
        df.loc[mask, 'eccentricity'] = (df.loc[mask, 'aphelion_dist'] / 
                                         df.loc[mask, 'semi_major_axis']) - 1
        print(f"✓ Physics: Calculated {mask.sum()} eccentricity values")
    
    # 3. Fill aphelion using eccentricity
    if 'eccentricity' in df.columns:
        mask = (df['aphelion_dist'].isna() & 
                df['semi_major_axis'].notna() & 
                df['eccentricity'].notna())
        if mask.sum() > 0:
            df.loc[mask, 'aphelion_dist'] = (df.loc[mask, 'semi_major_axis'] * 
                                              (1 + df.loc[mask, 'eccentricity']))
            print(f"✓ Physics: Filled {mask.sum()} aphelion_dist")
    
    # 4. Temporal features - simple forward fill
    temporal_cols = ['epoch_osculation', 'perihelion_time', 'epoch_date_close_approach']
    for col in temporal_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    for col in ['approach_year', 'approach_month', 'approach_day']:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].median())
    
    return df

X_train = physics_based_imputation_v2(X_train)
X_test = physics_based_imputation_v2(X_test)

# ============================================
# STEP 4: Angular Features → Sin/Cos (BEFORE MICE)
# ============================================
# ChatGPT is right: Transform FIRST, then impute

def convert_angles_to_circular(df):
    """Convert angular features to sin/cos representation"""
    angular_cols = ['asc_node_longitude', 'perihelion_arg', 'mean_anomaly']
    
    for col in angular_cols:
        if col in df.columns:
            radians = np.deg2rad(df[col])
            df[f'{col}_sin'] = np.sin(radians)
            df[f'{col}_cos'] = np.cos(radians)
    
    # Drop original degree columns
    df = df.drop(columns=[c for c in angular_cols if c in df.columns])
    
    return df

X_train = convert_angles_to_circular(X_train)
X_test = convert_angles_to_circular(X_test)

print("\n✓ Converted angular features to sin/cos")

# ============================================
# STEP 5: Handle Categoricals
# ============================================
# ChatGPT's advice: Missing as its own category

categorical_cols = ['relative_velocity_km_per_sec', 'orbital_period', 'orbit_uncertainity']

for col in categorical_cols:
    if col in X_train.columns:
        # Fill missing with "Unknown" category
        X_train[col] = X_train[col].fillna('Unknown')
        X_test[col] = X_test[col].fillna('Unknown')
        
        # Convert to category type for MICE
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

print("✓ Handled categorical features")

# ============================================
# STEP 6: MICE for Remaining Features
# ============================================
# Now MICE will impute:
# - sin/cos components (preserving correlations!)
# - miss_dist_astronomical (correlated with velocity, hazard)
# - miles_per_hour (correlated with orbit, hazard)
# - jupiter_tisserand_invariant (if still missing)

print("\nRemaining missing values before MICE:")
missing = X_train.isna().sum()
print(missing[missing > 0])

# Reset index for MICE
X_train_mice = X_train.copy().reset_index(drop=True)
X_test_mice = X_test.copy().reset_index(drop=True)

# Fit MICE on training data
print("\nFitting MICE on training data...")
kds = mf.ImputationKernel(
    X_train_mice,
    random_state=42,
    mean_match_candidates=5  # Prevents KDTree errors
)

kds.mice(iterations=5, verbose=True)
X_train_imputed = kds.complete_data()

# Apply to test data
print("\nImputing test data...")
X_test_imputed = kds.impute_new_data(X_test_mice).complete_data()

# ============================================
# STEP 7: Post-MICE Renormalization (Optional)
# ============================================
# ChatGPT suggested this: Enforce sin² + cos² = 1

def renormalize_circular(df):
    """Ensure sin² + cos² = 1 for circular features"""
    angular_bases = ['asc_node_longitude', 'perihelion_arg', 'mean_anomaly']
    
    for base in angular_bases:
        sin_col = f'{base}_sin'
        cos_col = f'{base}_cos'
        
        if sin_col in df.columns and cos_col in df.columns:
            # Calculate magnitude
            magnitude = np.sqrt(df[sin_col]**2 + df[cos_col]**2)
            
            # Renormalize
            df[sin_col] = df[sin_col] / magnitude
            df[cos_col] = df[cos_col] / magnitude
    
    return df

X_train_final = renormalize_circular(X_train_imputed)
X_test_final = renormalize_circular(X_test_imputed)

print("\n✓ Renormalized circular features")

# ============================================
# STEP 8: Validation
# ============================================

print("\n" + "="*60)
print("FINAL VALIDATION")
print("="*60)

# Check no missing values
print(f"\nMissing in train: {X_train_final.isna().sum().sum()}")
print(f"Missing in test: {X_test_final.isna().sum().sum()}")

# Validate Kepler's 3rd Law still holds
AU = 1.496e11
G = 6.67430e-11
M_SUN = 1.989e30

mask = X_train_final['semi_major_axis'].notna() & X_train_final['mean_motion'].notna()
if mask.sum() > 0:
    a_m = X_train_final.loc[mask, 'semi_major_axis'] * AU
    n_calc = np.sqrt(G * M_SUN / a_m**3) * (180/np.pi) * 86400
    n_actual = X_train_final.loc[mask, 'mean_motion']
    
    error = np.abs(n_calc - n_actual) / n_actual * 100
    print(f"\nKepler's Law validation:")
    print(f"  Mean error: {error.mean():.4f}%")
    print(f"  Max error: {error.max():.4f}%")

# Validate circular features
print("\nCircular feature validation (sin² + cos² = 1):")
for base in ['asc_node_longitude', 'perihelion_arg', 'mean_anomaly']:
    sin_col = f'{base}_sin'
    cos_col = f'{base}_cos'
    
    if sin_col in X_train_final.columns:
        magnitude = X_train_final[sin_col]**2 + X_train_final[cos_col]**2
        error = np.abs(magnitude - 1)
        print(f"  {base}: max error = {error.max():.2e}")

print("\n✅ Data ready for modeling!")
# Constants
AU = 1.496e11
G = 6.67430e-11
M_SUN = 1.989e30

# Calculate what mean_motion SHOULD be
a_m = X_train_final['semi_major_axis'] * AU
n_calculated = np.sqrt(G * M_SUN / a_m**3) * (180/np.pi) * 86400
n_actual = X_train_final['mean_motion']

# Calculate error
relative_error = np.abs(n_calculated - n_actual) / n_actual * 100

# Find the worst offenders
worst_indices = relative_error.nlargest(10).index

print("Top 10 asteroids violating Kepler's 3rd Law:")
print("="*80)
suspect_df = pd.DataFrame({
    'semi_major_axis': X_train_final.loc[worst_indices, 'semi_major_axis'],
    'mean_motion_actual': n_actual.loc[worst_indices],
    'mean_motion_expected': n_calculated.loc[worst_indices],
    'error_%': relative_error.loc[worst_indices]
})
print(suspect_df)

# Check if these were imputed or original
print("\nWere these values imputed by MICE?")
print("(Check if they had missing semi_major_axis or mean_motion originally)")

# Distribution of errors
print("\nError distribution:")
print(f"< 5% error: {(relative_error < 5).sum()} asteroids ({(relative_error < 5).sum()/len(relative_error)*100:.1f}%)")
print(f"5-10% error: {((relative_error >= 5) & (relative_error < 10)).sum()} asteroids")
print(f"10-20% error: {((relative_error >= 10) & (relative_error < 20)).sum()} asteroids")
print(f"> 20% error: {(relative_error >= 20).sum()} asteroids ({(relative_error >= 20).sum()/len(relative_error)*100:.1f}%)")
# Check which asteroids originally had missing values
# (You'll need to run this on your ORIGINAL data before imputation)

# Load original data again
data_original = pd.read_csv('train.csv')
X_original = data_original.drop(columns=['hazardous'])

X_train_original, _, _, _ = train_test_split(
    X_original, data_original['hazardous'], 
    test_size=0.2, 
    random_state=42,
    stratify=data_original['hazardous']
)

# Check the problem indices
problem_indices = [2104, 1609, 391, 466, 102, 2819, 2587, 2559, 2879]

print("Did these asteroids have missing values originally?")
print("="*80)
for idx in problem_indices:
    sma_missing = pd.isna(X_train_original.loc[idx, 'semi_major_axis'])
    mm_missing = pd.isna(X_train_original.loc[idx, 'mean_motion'])
    print(f"Index {idx}: semi_major_axis missing={sma_missing}, mean_motion missing={mm_missing}")


# ============================================
# STEP 1: CREATE VIOLATION FEATURE (OPTIONAL)
# ============================================

def add_kepler_violation_feature(df):
    """
    Capture magnitude of Kepler's law violation
    This might indicate measurement uncertainty
    """
    df = df.copy()
    
    AU = 1.496e11
    G = 6.67430e-11
    M_SUN = 1.989e30
    
    a_m = df['semi_major_axis'] * AU
    n_expected = np.sqrt(G * M_SUN / a_m**3) * (180/np.pi) * 86400
    
    # Relative error as a feature
    df['kepler_violation'] = np.abs(
        df['mean_motion'] - n_expected
    ) / df['mean_motion']
    
    print(f"✓ Created kepler_violation feature")
    print(f"  Mean violation: {df['kepler_violation'].mean()*100:.2f}%")
    print(f"  Max violation: {df['kepler_violation'].max()*100:.2f}%")
    
    return df

X_train_final = add_kepler_violation_feature(X_train_final)
X_test_final = add_kepler_violation_feature(X_test_final)

# ============================================
# STEP 2: ENFORCE KEPLER'S LAW
# ============================================

def enforce_keplers_law(df):
    """
    Recalculate mean_motion from semi_major_axis
    """
    df = df.copy()
    
    AU = 1.496e11
    G = 6.67430e-11
    M_SUN = 1.989e30
    
    a_m = df['semi_major_axis'] * AU
    df['mean_motion'] = np.sqrt(G * M_SUN / a_m**3) * (180/np.pi) * 86400
    
    return df

X_train_final = enforce_keplers_law(X_train_final)
X_test_final = enforce_keplers_law(X_test_final)

print("\n✓ Physics enforced: Kepler's 3rd Law now holds")

# ============================================
# STEP 3: VALIDATE
# ============================================

# Should now be perfect
a_m = X_train_final['semi_major_axis'] * AU
n_calc = np.sqrt(G * M_SUN / a_m**3) * (180/np.pi) * 86400
error = np.abs(n_calc - X_train_final['mean_motion']) / X_train_final['mean_motion'] * 100

print(f"\nValidation:")
print(f"  Mean error: {error.mean():.2e}%")  # Should be ~0
print(f"  Max error: {error.max():.2e}%")    # Should be ~0
# Quick visual check: Do hazardous asteroids have different orbital energies?
mask_ = X_train_final['kepler_violation'] * 100 > 10
X_train_final.loc[mask_, 'kepler_violation'] * 100
violations = X_train_final.loc[
    X_train_final['kepler_violation'] * 100 > 1,
    'kepler_violation'
] * 100
print(len(X_train_final))
print(violations)

plt.scatter(X_train_final['kepler_violation'], y_train)
plt.show()
missing_counts_2 = X_train_final.isnull().sum()
missing_percentage_2 = (X_train_final.isnull().mean() * 100)

# Combine into a summary table for easy viewing
missing_summary_2 = pd.DataFrame({
    'Missing Count': missing_counts_2,
    'Percentage (%)': missing_percentage_2,
    'dtype': X_train_final.dtypes
}).sort_values(by='Percentage (%)', ascending=False)

print(missing_summary_2)

columns_to_drop = ['relative_velocity_km_per_sec', 'orbit_uncertainity', 'orbital_period']
X_train_num =  X_train_final.drop(columns = columns_to_drop)
X_train_obj = X_train_final.select_dtypes(include=['object'])
X_test_num =  X_test_final.drop(columns = columns_to_drop)
X_test_obj = X_test_final.select_dtypes(include=['object'])
print("only data with numbers")
print(X_train_num.info())
print("only data with objects")
print(X_train_obj.info())

skewness_vals = X_train_num.skew().sort_values(ascending=False)

# Flag features that need transformation (> 0.5 or < -0.5)
transform_needed = skewness_vals
print("output of the .skew")
print(transform_needed)