# Machine Learning Algorithm Experiment Results

This experiment evaluates the effectiveness of various machine learning algorithms on tabular data. The dataset was preprocessed, and different classifiers were implemented. Accuracy scores were obtained for each model, and further tuning was applied to the CatBoost and MLP models.

---

## Data Preprocessing

- **Data Size**: 78 samples, 18 features
- **Training Set**: 62 samples
- **Test Set**: 16 samples
- **Features**: All columns except 'Patient'
- **Target**: 'Patient'

---

## Model Performance

### 1. Random Forest
- **Accuracy**: 0.6875

### 2. XGBoost
- **Accuracy**: 0.5625

### 3. CatBoost
- **Accuracy**: 0.6875

### 4. MLP (Sklearn)
- **Accuracy**: 0.6250

---

## Hyperparameter Tuning

### Grid Search for CatBoost
- **Best Parameters**: `{'border_count': 32, 'depth': 4, 'iterations': 100, 'l2_leaf_reg': 5, 'learning_rate': 0.01}`
- **Best Accuracy on CV**: 0.7579
- **Test Set Accuracy with Best Parameters**: 0.8125

### Random Search for CatBoost
- **Best Parameters**: `{'random_strength': 17.89, 'learning_rate': 0.171, 'l2_leaf_reg': 5, 'iterations': 100, 'depth': 5, 'border_count': 128, 'bagging_temperature': 0.0}`
- **Best Accuracy on CV**: 0.7746
- **Test Set Accuracy with Best Parameters**: 0.6250

---

## MLP using PyTorch

- **Epochs**: 50
- **Loss at Final Epoch**: 0.4056
- **MLP Model Accuracy (PyTorch)**: 0.7500

---

## Summary

| Model              | Accuracy     | Tuning Method        | Best Parameters                                                                                 |
|--------------------|--------------|----------------------|------------------------------------------------------------------------------------------------|
| Random Forest      | 0.6875       | None                 | -                                                                                              |
| XGBoost            | 0.5625       | None                 | -                                                                                              |
| CatBoost (Grid)    | 0.8125       | Grid Search          | `{'border_count': 32, 'depth': 4, 'iterations': 100, 'l2_leaf_reg': 5, 'learning_rate': 0.01}` |
| CatBoost (Random)  | 0.6250       | Random Search        | `{'random_strength': 17.89, 'learning_rate': 0.171, 'l2_leaf_reg': 5, 'iterations': 100, 'depth': 5, 'border_count': 128, 'bagging_temperature': 0.0}` |
| MLP (Sklearn)      | 0.6250       | None                 | -                                                                                              |
| MLP (PyTorch)      | 0.7500       | None                 | -                                                                                              |

Best accuracy 0.81 was aqquired using CatBoost.
