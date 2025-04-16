import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../statisticalinf/finaldata_onehotafterdropreefname.csv')
print("Dataset loaded successfully!")
print("\nFirst few rows of the dataset:")
print(df.head())

# Prepare the data
X = df.drop(['Bleaching'], axis=1)
y = df['Bleaching']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Initialize models
xgb_model = xgb.XGBClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
lgb_model = lgb.LGBMClassifier(random_state=42)

# Evaluate models
results = []

for model, name in [(xgb_model, 'XGBoost'), (rf_model, 'Random Forest'), (lgb_model, 'LightGBM')]:
    print(f"\nEvaluating {name} model...")
    result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    results.append(result)

# Create results dataframe
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df)

# Plot comparison of models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(12, 6))
for i, model in enumerate(['XGBoost', 'Random Forest', 'LightGBM']):
    values = results_df[results_df['Model'] == model][metrics].values[0]
    plt.bar(x + i*width, values, width, label=model)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.xticks(x + width, metrics)
plt.legend()
plt.tight_layout()
plt.show() 