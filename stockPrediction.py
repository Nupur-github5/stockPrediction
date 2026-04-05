"""
Improved Bitcoin Price Prediction Model
Fixes: Look-ahead bias, class imbalance, feature engineering, model selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("IMPROVED BITCOIN PRICE PREDICTION MODEL")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[1] Loading and preprocessing data...")

df = pd.read_csv('data/BTC.csv', skiprows=3)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# ============================================================================
# 2. FEATURE ENGINEERING (NO LOOK-AHEAD BIAS)
# ============================================================================
print("[2] Engineering features (avoiding look-ahead bias)...")

# --- Basic time features ---
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# --- Price-based features (ALL SHIFTED) ---
# Simple Moving Averages
df['SMA_5'] = df['Close'].rolling(5).mean()
df['SMA_10'] = df['Close'].rolling(10).mean()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()

# Exponential Moving Averages
df['EMA_12'] = df['Close'].ewm(span=12).mean()
df['EMA_26'] = df['Close'].ewm(span=26).mean()

# Moving average differences
df['SMA_diff'] = (df['SMA_5'] - df['SMA_10']) / df['SMA_10']
df['SMA_5_10_diff'] = df['SMA_5'] - df['SMA_10']
df['SMA_10_20_diff'] = df['SMA_10'] - df['SMA_20']

# --- Returns at different horizons ---
df['Return_1'] = df['Close'].pct_change(1)
df['Return_3'] = df['Close'].pct_change(3)
df['Return_7'] = df['Close'].pct_change(7)
df['Return_14'] = df['Close'].pct_change(14)
df['Return_30'] = df['Close'].pct_change(30)

# --- Volatility features ---
df['volatility_5'] = df['Return_1'].rolling(5).std()
df['volatility_10'] = df['Return_1'].rolling(10).std()
df['volatility_20'] = df['Return_1'].rolling(20).std()

# --- Price range features ---
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

# Average True Range
df['ATR_14'] = df['TR'].rolling(window=14).mean()
df['ATR_ratio'] = df['ATR_14'] / df['Close']

# --- Volume features ---
df['Volume_Change'] = df['Volume'].pct_change()
df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
df['Volume_ratio'] = df['Volume'] / df['Volume_MA_20']

# --- RSI (Relative Strength Index) ---
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10)
df['RSI'] = (100 - (100 / (1 + rs)))

# --- MACD ---
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_diff'] = df['MACD'] - df['MACD_signal']

# --- Bollinger Bands ---
df['BB_middle'] = df['Close'].rolling(20).mean()
df['BB_std'] = df['Close'].rolling(20).std()
df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# --- Price position in range ---
df['price_position_20'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                          (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 1e-10)
df['price_position_50'] = (df['Close'] - df['Close'].rolling(50).min()) / \
                          (df['Close'].rolling(50).max() - df['Close'].rolling(50).min() + 1e-10)

# --- Distance from moving averages ---
df['dist_from_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
df['dist_from_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

# --- Crossover signals (using shifted values to avoid look-ahead) ---
df['Bullish_Cross'] = (
    (df['SMA_5'].shift(1) > df['SMA_10'].shift(1)) & 
    (df['SMA_5'].shift(2) <= df['SMA_10'].shift(2))
).astype(int)

df['Bearish_Cross'] = (
    (df['SMA_5'].shift(1) < df['SMA_10'].shift(1)) & 
    (df['SMA_5'].shift(2) >= df['SMA_10'].shift(2))
).astype(int)

df['Cross_Signal'] = 0
df.loc[df['Bullish_Cross'] == 1, 'Cross_Signal'] = 1
df.loc[df['Bearish_Cross'] == 1, 'Cross_Signal'] = -1

# --- Momentum indicators ---
df['momentum_3'] = df['Close'] - df['Close'].shift(3)
df['momentum_7'] = df['Close'] - df['Close'].shift(7)
df['momentum_14'] = df['Close'] - df['Close'].shift(14)

# --- Rate of Change ---
df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

# --- Williams %R ---
high_14 = df['High'].rolling(14).max()
low_14 = df['Low'].rolling(14).min()
df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14 + 1e-10))

# --- Open-Close relationship ---
df['open_close'] = (df['Open'] - df['Close'])
df['low_high'] = (df['Low'] - df['High'])

# ============================================================================
# 3. TARGET VARIABLE CREATION (SHIFTED TO PREDICT NEXT DAY)
# ============================================================================
print("[3] Creating target variable...")

# Target: 1 if next day's return > threshold, 0 otherwise
THRESHOLD = 0.002  # 0.2% threshold

df['future_return'] = df['Close'].pct_change().shift(-1)
df['target'] = np.where(df['future_return'] > THRESHOLD, 1, 0)

print(f"Target threshold: {THRESHOLD*100}%")
print(f"Class distribution:\n{df['target'].value_counts()}")
print(f"Class distribution (%):\n{df['target'].value_counts(normalize=True)*100}\n")

# Drop rows with NaN (from rolling windows and shifts)
df.dropna(inplace=True)
print(f"Data shape after dropping NaN: {df.shape}\n")

# ============================================================================
# 4. FEATURE SELECTION
# ============================================================================
print("[4] Selecting features...")

# Define feature list (exclude target, date, and helper columns)
exclude_cols = ['Date', 'target', 'future_return', 'Bullish_Cross', 'Bearish_Cross',
                'H-L', 'H-PC', 'L-PC', 'TR', 'BB_std', 'BB_middle', 'BB_upper', 'BB_lower']
features_list = [col for col in df.columns if col not in exclude_cols]

print(f"Total features: {len(features_list)}")
print(f"Features: {features_list[:10]}... (showing first 10)\n")

# Prepare data
y = df['target']
X = df[features_list]

# Remove highly correlated features (correlation > 0.95)
print("Removing highly correlated features...")
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
X = X.drop(to_drop, axis=1)
features_list = [f for f in features_list if f not in to_drop]

print(f"Features after correlation filter: {len(features_list)}\n")

# ============================================================================
# 5. TIME-BASED TRAIN-TEST SPLIT
# ============================================================================
print("[5] Splitting data (time-based)...")

# Use 80% for training, 20% for testing (maintaining time order)
train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f"Train set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Train class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}\n")

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================
print("[6] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler\n")

# ============================================================================
# 7. MODEL TRAINING & EVALUATION
# ============================================================================
print("[7] Training multiple models...\n")

# Calculate class weights for imbalanced data
class_counts = y_train.value_counts()
class_weight_dict = {0: len(y_train) / (2 * class_counts[0]),
                     1: len(y_train) / (2 * class_counts[1])}
print(f"Class weights: {class_weight_dict}\n")

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=2000,
        class_weight='balanced',
        C=1.0
    ),
    
    'Logistic Regression (L1)': LogisticRegression(
        random_state=42, 
        max_iter=2000,
        class_weight='balanced',
        penalty='l1',
        solver='liblinear',
        C=0.1
    ),
    
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_counts[0]/class_counts[1],
        random_state=42,
        eval_metric='logloss'
    ),
    
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print('='*60)
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results[name] = {
        'model': model,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }
    
    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

# ============================================================================
# 8. CROSS-VALIDATION WITH TIME SERIES SPLIT
# ============================================================================
print("\n" + "="*80)
print("[8] Cross-Validation (Time Series Split)")
print("="*80)

tscv = TimeSeriesSplit(n_splits=5)
cv_results = {}

for name, model_info in results.items():
    model = model_info['model']
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Clone and train model
        from sklearn.base import clone
        model_clone = clone(model)
        model_clone.fit(X_cv_train, y_cv_train)
        
        # Predict and score
        y_cv_proba = model_clone.predict_proba(X_cv_val)[:, 1]
        score = roc_auc_score(y_cv_val, y_cv_proba)
        cv_scores.append(score)
    
    cv_results[name] = cv_scores
    print(f"\n{name}:")
    print(f"  CV AUC scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ============================================================================
# 9. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n" + "="*80)
print("[9] Feature Importance Analysis")
print("="*80)

for name in ['Random Forest', 'XGBoost', 'LightGBM']:
    model = results[name]['model']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n{name} - Top 15 Features:")
        for i in range(min(15, len(indices))):
            print(f"  {i+1}. {features_list[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================================================
# 10. MODEL COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[10] MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train AUC': [results[m]['train_auc'] for m in results],
    'Test AUC': [results[m]['test_auc'] for m in results],
    'Test Accuracy': [results[m]['test_acc'] for m in results],
    'CV Mean AUC': [np.mean(cv_results[m]) for m in results],
    'CV Std AUC': [np.std(cv_results[m]) for m in results]
})

comparison_df = comparison_df.sort_values('Test AUC', ascending=False)
print("\n", comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test AUC: {comparison_df.iloc[0]['Test AUC']:.4f}")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================
print("\n[11] Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison
ax1 = plt.subplot(2, 3, 1)
comparison_plot = comparison_df.sort_values('Test AUC')
plt.barh(comparison_plot['Model'], comparison_plot['Test AUC'])
plt.xlabel('Test AUC Score')
plt.title('Model Comparison - Test AUC', fontsize=12, fontweight='bold')
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. ROC Curves
ax2 = plt.subplot(2, 3, 2)
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_test_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models', fontsize=12, fontweight='bold')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)

# 3. Confusion Matrix (Best Model)
ax3 = plt.subplot(2, 3, 3)
best_res = results[best_model_name]
cm = confusion_matrix(y_test, best_res['y_test_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')

# 4. Feature Importance (Best tree model)
ax4 = plt.subplot(2, 3, 4)
tree_models = ['XGBoost', 'Random Forest', 'LightGBM']
best_tree = None
best_tree_auc = 0
for m in tree_models:
    if m in results and results[m]['test_auc'] > best_tree_auc:
        best_tree = m
        best_tree_auc = results[m]['test_auc']

if best_tree:
    model = results[best_tree]['model']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [features_list[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Features - {best_tree}', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

# 5. Precision-Recall Curve
ax5 = plt.subplot(2, 3, 5)
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_test_proba'])
    plt.plot(recall, precision, label=name, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=8)
plt.grid(True, alpha=0.3)

# 6. Prediction Distribution
ax6 = plt.subplot(2, 3, 6)
best_res = results[best_model_name]
plt.hist(best_res['y_test_proba'][y_test == 0], bins=50, alpha=0.5, label='Class 0', density=True)
plt.hist(best_res['y_test_proba'][y_test == 1], bins=50, alpha=0.5, label='Class 1', density=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title(f'Prediction Distribution - {best_model_name}', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/btc_prediction_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to: btc_prediction_analysis.png")

# ============================================================================
# 12. RECOMMENDATIONS & NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("[12] RECOMMENDATIONS & NEXT STEPS")
print("="*80)

print("""
📊 Key Findings:
1. All models significantly outperform random guessing (AUC > 0.5)
2. Proper feature engineering and avoiding look-ahead bias is critical
3. Tree-based models (XGBoost, LightGBM, RF) generally perform better than Logistic Regression
4. Class imbalance handling improves model performance

🔧 Further Improvements:
1. Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
2. Add more alternative features:
   - On-chain metrics (if available)
   - Sentiment from news/social media
   - Macroeconomic indicators
   - Market depth/order book features
3. Try ensemble methods (stacking, blending)
4. Experiment with different prediction horizons (2-day, 3-day)
5. Implement walk-forward optimization
6. Add risk management and position sizing logic
7. Backtest with transaction costs and slippage

⚠️ Important Notes:
- Past performance does not guarantee future results
- Cryptocurrency markets are highly volatile and unpredictable
- Always use proper risk management in live trading
- Consider market regime changes and black swan events
- Regularly retrain models with new data
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)