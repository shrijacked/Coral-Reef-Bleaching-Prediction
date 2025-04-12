import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('../statisticalinf/finaldata_onehotafterdropreefname.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Bleaching distribution:\n{df['Bleaching'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum().sum()}")

# Define features and target
y = df['Bleaching']
X = df.drop('Bleaching', axis=1)

# Handle categorical columns
# The data already appears to be one-hot encoded for categorical variables
# But let's make sure to exclude any non-numeric columns
categorical_columns = []
for col in X.columns:
    if X[col].dtype == 'object':
        categorical_columns.append(col)

if categorical_columns:
    print(f"Categorical columns found: {categorical_columns}")
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
else:
    print("No categorical columns found, data already processed")

# Feature selection - Year and Month might not be directly relevant
# But we'll keep them for now and let the model decide
non_feature_cols = []
if non_feature_cols:
    X = X.drop(non_feature_cols, axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train_scaled, label=y_train)
test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

# Parameters for LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Train LightGBM model
print("Training LightGBM model...")
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Make predictions
y_pred_proba = gbm.predict(X_test_scaled, num_iteration=gbm.best_iteration)
y_pred = np.round(y_pred_proba)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Bleaching', 'Bleaching'], 
            yticklabels=['No Bleaching', 'Bleaching'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Feature importance
plt.figure(figsize=(12, 8))
feature_importance = gbm.feature_importance(importance_type='split')
feature_names = list(X.columns)
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance (Split)')
plt.xlabel('Number of splits')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Cross-validation for more robust model evaluation
print("\nPerforming 5-fold cross-validation...")
cv_results = {}
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scale the features
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)
    
    # Create dataset for LightGBM
    train_data_fold = lgb.Dataset(X_train_fold_scaled, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold_scaled, label=y_val_fold, reference=train_data_fold)
    
    # Train model
    gbm_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=1000,
        valid_sets=[train_data_fold, val_data_fold],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
        verbose_eval=False
    )
    
    # Make predictions
    y_pred_fold_proba = gbm_fold.predict(X_val_fold_scaled, num_iteration=gbm_fold.best_iteration)
    y_pred_fold = np.round(y_pred_fold_proba)
    
    # Store metrics
    accuracy = accuracy_score(y_val_fold, y_pred_fold)
    auc = roc_auc_score(y_val_fold, y_pred_fold_proba)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    cv_results[f'fold_{fold + 1}'] = {
        'accuracy': accuracy,
        'auc': auc
    }

# Calculate average CV metrics
cv_accuracy = np.mean([cv_results[f]['accuracy'] for f in cv_results])
cv_auc = np.mean([cv_results[f]['auc'] for f in cv_results])
print(f"\nAverage CV Accuracy: {cv_accuracy:.4f}")
print(f"Average CV AUC: {cv_auc:.4f}")

# Save the model
gbm.save_model('coral_bleaching_lgbm_model.txt')
print("\nModel saved to coral_bleaching_lgbm_model.txt")

print("\nProcess completed!") 