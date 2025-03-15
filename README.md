# Breast-cancer-prediction

## Table of Contents
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Feature Importance](#feature-importance)

---

## Dataset
The **Breast Cancer Wisconsin (Diagnostic) Dataset** contains data from digitized images of breast mass cell nuclei in fine needle aspirates. The task is to classify the cells as either malignant (cancerous) or benign (non-cancerous) based on the features provided.

The dataset was fetched from the **UC Irvine Machine Learning Repository**.

---

## Model
The model uses **XGBoost** (`XGBClassifier`)

### 1. Data Preprocessing
- **Label Encoding**: The target variable (`Malignant` and `Benign`) is encoded into numerical values (1 and 0, respectively).
- **Train-Test Split**: The data is split into a training set (80%) and a test set (20%).
- **Correlation Handling**: Highly correlated features (correlation > 0.9) are dropped to avoid multicollinearity.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to handle class imbalance by oversampling the minority class in the training set.

### 2. Feature Selection
- **RFE (Recursive Feature Elimination)**: A feature selection method used to select the most important 20 features that contribute the most to the model's performance.

### 3. Model Tuning
- **GridSearchCV**: Hyperparameter tuning is performed using GridSearchCV to find the best parameters for the XGBoost classifier. The grid includes:
  - `learning_rate`, `n_estimators`, `max_depth`, `gamma`, `reg_alpha`, `reg_lambda`, and `scale_pos_weight`.

### 4. Evaluation
- **Classification Report**: The model's performance is evaluated using precision, recall, F1-score, and support.
- **ROC-AUC Score**: The area under the ROC curve is calculated to assess the model's performance in distinguishing between the two classes.

---

## Results
The best model, after hyperparameter tuning, is evaluated on the test set. Below is the classification report for the **Breast Cancer Dataset** model:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Benign) | 0.99      | 0.96   | 0.97     | 71      |
| 1 (Malignant) | 0.93      | 0.98   | 0.95     | 43      |
| **Accuracy** |           |        | **0.96** | **114** |
| **Macro avg** | 0.96      | 0.97   | 0.96     | 114     |
| **Weighted avg** | 0.97   | 0.96   | 0.97     | 114     |

### ROC-AUC Score:
The ROC-AUC score of the model is **0.9931**, indicating an excellent ability to distinguish between malignant and benign tumors with high confidence.
