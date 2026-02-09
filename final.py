"""
AZDC'26 Cosmic Collision Analysis - Complete Solution
Asteroid hazard classification using orbital mechanics and machine learning

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

import miceforest as mf
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler

print("Loading data...")

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['hazardous'].value_counts()}")

# Separate features and target
X = train.drop(columns=['hazardous'])
y = train['hazardous']
test_names = test['name'].copy()

# Physics-based imputation using orbital mechanics
def physics_imputation(df):
    """Fill missing values using orbital mechanics relationships"""
    df = df.copy()
    
    # Physical constants
    AU = 1.496e11  # Astronomical Unit in meters
    G = 6.67430e-11  # Gravitational constant
    M_SUN = 1.989e30  # Solar mass in kg
    
    # Fill mean_motion from semi_major_axis using Kepler's third law
    if 'mean_motion' in df.columns and 'semi_major_axis' in df.columns:
        mask = df['mean_motion'].isna() & df['semi_major_axis'].notna()
        if mask.any():
            a_m = df.loc[mask, 'semi_major_axis'] * AU
            n_rad_per_sec = np.sqrt(G * M_SUN / a_m**3)
            df.loc[mask, 'mean_motion'] = n_rad_per_sec * (180/np.pi) * 86400
    
    # Fill semi_major_axis from mean_motion
    if 'semi_major_axis' in df.columns and 'mean_motion' in df.columns:
        mask = df['semi_major_axis'].isna() & df['mean_motion'].notna()
        if mask.any():
            n_rad_per_sec = df.loc[mask, 'mean_motion'] * (np.pi/180) / 86400
            a_m = (G * M_SUN / n_rad_per_sec**2)**(1/3)
            df.loc[mask, 'semi_major_axis'] = a_m / AU
    
    # Fill eccentricity from aphelion distance
    if 'aphelion_dist' in df.columns and 'semi_major_axis' in df.columns:
        if 'eccentricity' not in df.columns:
            df['eccentricity'] = np.nan
        mask = df['eccentricity'].isna() & df['aphelion_dist'].notna() & df['semi_major_axis'].notna()
        if mask.any():
            df.loc[mask, 'eccentricity'] = (df.loc[mask, 'aphelion_dist'] / df.loc[mask, 'semi_major_axis']) - 1
    
    # Fill aphelion_dist from semi_major_axis and eccentricity
    if 'aphelion_dist' in df.columns and 'semi_major_axis' in df.columns and 'eccentricity' in df.columns:
        mask = df['aphelion_dist'].isna() & df['semi_major_axis'].notna() & df['eccentricity'].notna()
        if mask.any():
            df.loc[mask, 'aphelion_dist'] = df.loc[mask, 'semi_major_axis'] * (1 + df.loc[mask, 'eccentricity'])
    
    # Remove negative values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'hazardous':
            neg_mask = df[col] < 0
            if neg_mask.any():
                df.loc[neg_mask, col] = np.nan
    
    return df

print("\nApplying physics-based imputation...")
X = physics_imputation(X)
test = physics_imputation(test)

# Feature engineering
def create_features(df):
    """Create advanced features from orbital mechanics"""
    df = df.copy()
    
    # Physical constants
    AU = 1.496e11
    G = 6.67430e-11
    M_SUN = 1.989e30
    EARTH_PERIOD = 365.25
    
    # Eccentricity derived
    if 'aphelion_dist' in df.columns and 'semi_major_axis' in df.columns:
        df['eccentricity_derived'] = (df['aphelion_dist'] / df['semi_major_axis']) - 1
        df['eccentricity_derived'] = df['eccentricity_derived'].clip(0, 0.999)
    
    # Perihelion distance
    if 'semi_major_axis' in df.columns and 'eccentricity_derived' in df.columns:
        df['perihelion_dist'] = df['semi_major_axis'] * (1 - df['eccentricity_derived'])
    
    # Orbital period (Kepler's 3rd Law)
    if 'semi_major_axis' in df.columns:
        a_meters = df['semi_major_axis'] * AU
        T_seconds = 2 * np.pi * np.sqrt(a_meters**3 / (G * M_SUN))
        df['orbital_period_days'] = T_seconds / 86400
    
    # Mean motion
    if 'orbital_period_days' in df.columns:
        df['mean_motion_rad_day'] = 2 * np.pi / df['orbital_period_days']
    
    # Specific orbital energy
    if 'semi_major_axis' in df.columns:
        a_meters = df['semi_major_axis'] * AU
        df['specific_orbital_energy'] = -G * M_SUN / (2 * a_meters)
    
    # Specific angular momentum
    if 'semi_major_axis' in df.columns and 'eccentricity_derived' in df.columns:
        a_meters = df['semi_major_axis'] * AU
        df['specific_angular_momentum'] = np.sqrt(G * M_SUN * a_meters * (1 - df['eccentricity_derived']**2))
    
    # Velocity at perihelion
    if 'semi_major_axis' in df.columns and 'eccentricity_derived' in df.columns:
        a_meters = df['semi_major_axis'] * AU
        df['velocity_at_perihelion'] = np.sqrt((G * M_SUN / a_meters) * ((1 + df['eccentricity_derived']) / (1 - df['eccentricity_derived'])))
    
    # Velocity at aphelion
    if 'semi_major_axis' in df.columns and 'eccentricity_derived' in df.columns:
        a_meters = df['semi_major_axis'] * AU
        df['velocity_at_aphelion'] = np.sqrt((G * M_SUN / a_meters) * ((1 - df['eccentricity_derived']) / (1 + df['eccentricity_derived'])))
    
    # Synodic period
    if 'orbital_period_days' in df.columns:
        df['synodic_period'] = 1 / np.abs(1/df['orbital_period_days'] - 1/EARTH_PERIOD)
    
    # Miss distance in Earth radii
    if 'miss_dist_astronomical' in df.columns:
        df['miss_dist_earth_radii'] = df['miss_dist_astronomical'] * 23455
    
    # Energy ratio at perihelion
    if 'velocity_at_perihelion' in df.columns and 'perihelion_dist' in df.columns:
        kinetic = 0.5 * df['velocity_at_perihelion']**2
        potential = G * M_SUN / (df['perihelion_dist'] * AU)
        df['energy_ratio_perihelion'] = kinetic / potential
    
    # Temporal features
    if 'epoch_date_close_approach' in df.columns:
        df['approach_datetime'] = pd.to_datetime(df['epoch_date_close_approach'], unit='s', errors='coerce')
        df['approach_month'] = df['approach_datetime'].dt.month
        df['approach_day_of_year'] = df['approach_datetime'].dt.dayofyear
        df['approach_year'] = df['approach_datetime'].dt.year
        df['days_since_epoch'] = (df['approach_datetime'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / 86400
    
    # Orbit crossing features
    if 'perihelion_dist' in df.columns:
        df['crosses_earth_orbit'] = (df['perihelion_dist'] < 1.0).astype(int)
    
    if 'aphelion_dist' in df.columns:
        df['earth_in_orbit_range'] = ((df['perihelion_dist'] < 1.0) & (df['aphelion_dist'] > 1.0)).astype(int)
    
    # Hazard indicators
    if 'miss_dist_astronomical' in df.columns and 'relative_velocity' in df.columns:
        df['hazard_score'] = df['relative_velocity'] / (df['miss_dist_astronomical'] + 1e-10)
    
    if 'absolute_magnitude' in df.columns and 'miss_dist_astronomical' in df.columns:
        df['brightness_proximity'] = (30 - df['absolute_magnitude']) / (df['miss_dist_astronomical'] + 1e-10)
    
    # Size estimation from absolute magnitude
    if 'absolute_magnitude' in df.columns:
        df['estimated_diameter_km'] = 1329 / np.sqrt(0.25) * 10**(-0.2 * df['absolute_magnitude'])
    
    # Kinetic energy estimate
    if 'estimated_diameter_km' in df.columns and 'relative_velocity' in df.columns:
        volume = (4/3) * np.pi * (df['estimated_diameter_km'] * 500)**3
        mass = volume * 2000
        df['kinetic_energy'] = 0.5 * mass * (df['relative_velocity'] * 1000)**2
    
    # Hill sphere radius
    if 'semi_major_axis' in df.columns and 'estimated_diameter_km' in df.columns:
        mass_estimate = (4/3) * np.pi * (df['estimated_diameter_km'] * 500)**3 * 2000
        df['hill_sphere'] = df['semi_major_axis'] * (mass_estimate / (3 * M_SUN))**(1/3)
    
    # Tisserand parameter (relative to Earth)
    if 'semi_major_axis' in df.columns and 'eccentricity_derived' in df.columns and 'inclination' in df.columns:
        a_earth = 1.0
        inc_rad = df['inclination'] * np.pi / 180
        df['tisserand'] = (a_earth / df['semi_major_axis']) + 2 * np.sqrt(df['semi_major_axis'] / a_earth * (1 - df['eccentricity_derived']**2)) * np.cos(inc_rad)
    
    return df

print("Engineering features...")
X = create_features(X)
test = create_features(test)

# MICE imputation for remaining missing values
print("\nApplying MICE imputation for remaining missing values...")
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if 'hazardous' in numerical_cols:
    numerical_cols.remove('hazardous')

if X[numerical_cols].isnull().sum().sum() > 0:
    kernel = mf.ImputationKernel(X[numerical_cols], save_all_iterations=False, random_state=42)
    kernel.mice(3, verbose=False)
    X[numerical_cols] = kernel.complete_data()

if test[numerical_cols].isnull().sum().sum() > 0:
    kernel_test = mf.ImputationKernel(test[numerical_cols], save_all_iterations=False, random_state=42)
    kernel_test.mice(3, verbose=False)
    test[numerical_cols] = kernel_test.complete_data()

# Handle infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)
test.fillna(test.median(), inplace=True)

# Feature scaling
print("Scaling features...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Hyperparameter optimization with Optuna
print("\nRunning hyperparameter optimization...")

def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    
    # Model selection
    model_type = trial.suggest_categorical('model', ['XGBoost', 'LightGBM', 'CatBoost', 'BalancedRF'])
    
    # Imbalance handling
    imbalance_method = trial.suggest_categorical('imbalance', ['none', 'class_weight', 'SMOTE', 'undersample'])
    
    # Model-specific parameters
    if model_type == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        if imbalance_method == 'class_weight':
            scale = (y == 0).sum() / (y == 1).sum()
            params['scale_pos_weight'] = scale
        
        model = xgb.XGBClassifier(**params)
    
    elif model_type == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42,
            'verbose': -1
        }
        
        if imbalance_method == 'class_weight':
            params['is_unbalance'] = True
        
        model = lgb.LGBMClassifier(**params)
    
    elif model_type == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': False
        }
        
        if imbalance_method == 'class_weight':
            params['auto_class_weights'] = 'Balanced'
        
        model = cb.CatBoostClassifier(**params)
    
    elif model_type == 'BalancedRF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42
        }
        model = BalancedRandomForestClassifier(**params)
    
    # Apply sampling if needed
    if imbalance_method == 'SMOTE':
        k_neighbors = min(trial.suggest_int('smote_k', 3, 10), (y == 1).sum() - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        model = Pipeline([('sampler', smote), ('classifier', model)])
    elif imbalance_method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        model = Pipeline([('sampler', rus), ('classifier', model)])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score)
    
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=f1_scorer, n_jobs=-1)
    
    return scores.mean()

# Run optimization
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5,
        interval_steps=1
    )
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest F1-score: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# Train final model
print("\nTraining final model...")

best_params = study.best_params.copy()
model_type = best_params.pop('model')
imbalance_method = best_params.pop('imbalance', 'none')

# Create final model
if model_type == 'XGBoost':
    if imbalance_method == 'class_weight':
        scale = (y == 0).sum() / (y == 1).sum()
        best_params['scale_pos_weight'] = scale
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    final_model = xgb.XGBClassifier(**best_params)

elif model_type == 'LightGBM':
    if imbalance_method == 'class_weight':
        best_params['is_unbalance'] = True
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    final_model = lgb.LGBMClassifier(**best_params)

elif model_type == 'CatBoost':
    if imbalance_method == 'class_weight':
        best_params['auto_class_weights'] = 'Balanced'
    best_params['random_state'] = 42
    best_params['verbose'] = False
    final_model = cb.CatBoostClassifier(**best_params)

elif model_type == 'BalancedRF':
    best_params['random_state'] = 42
    final_model = BalancedRandomForestClassifier(**best_params)

# Apply sampling if needed
if imbalance_method == 'SMOTE':
    k_neighbors = min(best_params.get('smote_k', 5), (y == 1).sum() - 1)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    final_model = Pipeline([('sampler', smote), ('classifier', final_model)])
elif imbalance_method == 'undersample':
    rus = RandomUnderSampler(random_state=42)
    final_model = Pipeline([('sampler', rus), ('classifier', final_model)])

# Train final model
final_model.fit(X_scaled, y)

# Cross-validation scores
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_scaled, y, cv=cv, scoring='f1')
print(f"\nCross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Generate predictions
print("\nGenerating predictions...")
y_pred = final_model.predict(test_scaled)
print(f"Predicted hazardous: {y_pred.sum()}")
print(f"Predicted non-hazardous: {(y_pred == 0).sum()}")

# Create submission file
submission = pd.DataFrame({
    'name': test_names,
    'hazardous': y_pred
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Submission summary:\n{submission['hazardous'].value_counts()}")

print("\nPipeline complete!")
print(f"Best model: {model_type}")
print(f"Imbalance handling: {imbalance_method}")
print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
