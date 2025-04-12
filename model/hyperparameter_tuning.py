import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('../statisticalinf/finaldata_onehotafterdropreefname.csv')

# Define features and target
y = df['Bleaching']
X = df.drop('Bleaching', axis=1)

# Handle categorical columns
categorical_columns = []
for col in X.columns:
    if X[col].dtype == 'object':
        categorical_columns.append(col)

if categorical_columns:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        'verbose': -1
    }
    
    # Implement cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train_scaled, y_train):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
            verbose_eval=False
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_fold_val, y_pred)
        cv_scores.append(auc)
    
    # Return the mean AUC score
    return np.mean(cv_scores)

# Create the Optuna study
print("Starting hyperparameter tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best parameters
best_params = study.best_params
best_params['objective'] = 'binary'
best_params['metric'] = 'binary_logloss'
best_params['verbose'] = -1

print("Best parameters found:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# Train the model with the best parameters
print("\nTraining the model with the best parameters...")
train_data = lgb.Dataset(X_train_scaled, label=y_train)
test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

best_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Evaluate the model
y_pred_proba = best_model.predict(X_test_scaled, num_iteration=best_model.best_iteration)
y_pred = np.round(y_pred_proba)

print("\nModel Performance with Optimized Parameters:")
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc:.4f}")

# Save the optimized model
best_model.save_model('optimized_coral_bleaching_model.txt')
print("Optimized model saved to optimized_coral_bleaching_model.txt")

# Plot optimization history
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(study.trials) + 1), [t.value for t in study.trials], marker='o')
plt.xlabel('Trial')
plt.ylabel('AUC Score')
plt.title('Optimization History')
plt.grid(True)
plt.savefig('optimization_history.png')

# Plot parameter importances
param_importances = optuna.visualization.plot_param_importances(study)
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Parameter Importances')
plt.tight_layout()
plt.savefig('parameter_importances.png')

print("\nHyperparameter tuning completed!") 