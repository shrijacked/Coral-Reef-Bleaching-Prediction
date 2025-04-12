# Coral Bleaching Prediction Model

This directory contains a LightGBM implementation for predicting coral bleaching events based on environmental and biological factors.

## Files

- `coral_bleaching_lgbm.py`: Main implementation of the LightGBM model for coral bleaching prediction
- `hyperparameter_tuning.py`: Script for optimizing model parameters using Optuna
- `coral_bleaching_lgbm_model.txt`: Saved model file (after running the main script)
- `optimized_coral_bleaching_model.txt`: Saved optimized model file (after running the tuning script)

## Running the Model

1. Install requirements:
```
pip install -r ../requirements.txt
```

2. Run the basic model:
```
python coral_bleaching_lgbm.py
```

3. Run hyperparameter tuning:
```
python hyperparameter_tuning.py
```

## Model Information

- **Task**: Binary classification to predict coral bleaching (0 = no bleaching, 1 = bleaching)
- **Features**: Environmental factors (SST, Salinity, pH, etc.) and coral species presence
- **Evaluation Metrics**: Accuracy, ROC AUC, Precision, Recall, F1-score

## Output Files

- `confusion_matrix.png`: Visualization of model predictions vs actual values
- `feature_importance.png`: Graph showing the importance of each feature
- `optimization_history.png`: Plot of optimization trials (from hyperparameter tuning)
- `parameter_importances.png`: Graph showing the importance of each hyperparameter 