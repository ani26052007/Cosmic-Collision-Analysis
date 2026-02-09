# 🌌 AZDC'26 Cosmic Collision Analysis 
[Competition link](https://www.kaggle.com/competitions/azdc_26)


---

## 📊 Problem Statement

Classify Near-Earth Objects (NEOs) as **hazardous** or **non-hazardous** based on their orbital characteristics and physical properties. The dataset contains orbital parameters like semi-major axis, eccentricity, inclination, and approach distance.

**Challenge:** Highly imbalanced dataset with complex physics-based relationships between features and significant missing data.

---

## 🎯 Approach & Solution

### 1. **Physics-Based Imputation**
Instead of standard statistical imputation, I leveraged **orbital mechanics** to fill missing values:
- Applied **Kepler's Third Law** to derive mean motion from semi-major axis
- Calculated eccentricity from aphelion distance relationships
- Used gravitational equations to maintain physical consistency

```python
# Example: Deriving mean motion from semi-major axis
n = sqrt(G * M_sun / a³)
```

### 2. **Advanced Feature Engineering (25+ Features)**
Created domain-specific features based on celestial mechanics:

**Orbital Dynamics:**
- Perihelion & aphelion distances
- Orbital period (Kepler's 3rd Law)
- Specific orbital energy
- Specific angular momentum
- Velocities at perihelion/aphelion

**Hazard Indicators:**
- Earth orbit crossing detection
- Kinetic energy estimates
- Tisserand parameter (orbit stability)
- Hazard score (velocity/distance ratio)
- Brightness-proximity index

**Temporal Features:**
- Approach date decomposition (month, day of year)
- Synodic period with Earth

### 3. **MICE Imputation**
Applied **Multiple Imputation by Chained Equations** for remaining missing values after physics-based imputation.

### 4. **Handling Class Imbalance**
Tested multiple strategies:
- SMOTE (Synthetic Minority Oversampling)
- Random Undersampling
- Class weight balancing
- Balanced Random Forest

### 5. **Model Optimization**
- **Hyperparameter tuning** with Optuna (100 trials)
- Tested: XGBoost, LightGBM, CatBoost, Balanced Random Forest
- 5-fold Stratified Cross-Validation
- Optimization metric: F1-score

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **ML Frameworks** | XGBoost, LightGBM, CatBoost, Scikit-learn |
| **Imbalance Handling** | imbalanced-learn (SMOTE, RandomUnderSampler) |
| **Imputation** | MICE Forest |
| **Hyperparameter Tuning** | Optuna |
| **Data Processing** | Pandas, NumPy |
| **Scaling** | RobustScaler |

---

## 📈 Results

- **Cross-Validation F1-Score:** 0.77404
- **Public Leaderboard:** 6th place
- **Private Leaderboard:** 🥉 **5th Place**

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Solution
```bash
python final.py
```

**Required Files:**
- `train.csv` - Training dataset
- `test.csv` - Test dataset

**Output:**
- `submission.csv` - Predictions for submission

---

## 📁 Project Structure

```
AZAD_DATA_COMP/
│
├── final.py                 # Complete solution pipeline
├── imputation.ipynb         # Imputation experiments
├── notes.ipynb              # EDA and feature engineering notes
├── train.csv                # Training data
├── test.csv                 # Test data
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔍 Key Insights

1. **Domain knowledge matters:** Physics-based imputation significantly outperformed statistical methods
2. **Feature engineering > Model complexity:** Well-engineered orbital mechanics features provided more value than complex ensemble methods
3. **Imbalance handling is critical:** SMOTE and class weighting were essential for the minority (hazardous) class
4. **Optuna saved time:** Automated hyperparameter search across multiple model types found optimal configurations efficiently

---

## 📚 Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
miceforest>=5.0.0
imbalanced-learn>=0.9.0
optuna>=3.0.0
```

---

## 🎓 Competition Details

- **Event:** AZDC'26 Kaggle Competition
- **Organizer:** College Hall seniors
- **Participants:** College students only
- **Evaluation Metric:** F1-Score
- **Dataset Split:** 20% Public LB / 80% Private LB

---

## 🏅 Achievement

- **Rank:** 5th out of 17 teams

---

## 💡 Future Improvements

- Ensemble stacking of top 3 models
- Deep learning approach with physics-informed neural networks
- More sophisticated feature interactions
- Threshold optimization for final predictions
- External data augmentation (NASA JPL datasets)

---

## 🤝 Contributing

This is a competition solution, but feel free to:
- Open issues for questions
- Suggest improvements
- Fork and experiment with the approach

---

## 🙏 Acknowledgments

- Competition organizers for the challenging dataset
- Kaggle community for inspiration
- My college seniors for support and brainstorming

---
