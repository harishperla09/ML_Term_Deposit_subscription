# Bank Marketing Campaign Classification

## Problem Statement

Predict whether a client will subscribe to a term deposit based on marketing campaign data (phone calls) from a Portuguese bank. This is a binary classification problem - clients either subscribe (Yes) or don't subscribe (No).

The goal is to help the bank optimize marketing strategy by identifying potential subscribers.

## Dataset Description

**Name:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**Citation:** S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 2014

**Statistics:**
- Instances: 41,188
- Features: 20
- Target: y (yes/no)
- Missing Values: None

### Features

**Client Data:**
- age, job, marital, education
- default, housing, loan

**Campaign Data:**
- contact, month, day_of_week
- duration, campaign
- pdays, previous, poutcome

**Economic Indicators:**
- emp.var.rate, cons.price.idx, cons.conf.idx
- euribor3m, nr.employed

**Class Distribution:**
- No: 36,548 (88.7%)
- Yes: 4,640 (11.3%)
- The dataset is imbalanced

## Models Used

All models trained on 80-20 train-test split with stratified sampling.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.9147 | 0.8920 | 0.6347 | 0.5711 | 0.6012 | 0.5546 |
| kNN | 0.9053 | 0.8617 | 0.6267 | 0.3944 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest (Ensemble) | 0.9202 | 0.9515 | 0.7611 | 0.4256 | 0.5460 | 0.5318 |
| XGBoost (Ensemble) | 0.9227 | 0.9549 | 0.6907 | 0.5679 | 0.6233 | 0.5841 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Good baseline with 91.39% accuracy. High precision but low recall means it's conservative in predictions. Works well as a linear classifier for this dataset. |
| Decision Tree | Balanced precision-recall with F1 of 0.60. Parameter tuning with max_depth=10 prevented overfitting. Good interpretability but moderate performance. |
| kNN | Lowest performer with 90.53% accuracy due to curse of dimensionality. The high-dimensional feature space (20 features) makes distance-based classification less effective. Not suitable for this dataset. |
| Naive Bayes | Fastest training time but lowest accuracy (85.36%). The independence assumption between features is violated in this real-world dataset. Highest recall (0.6175) makes it useful when false negatives are costly. |
| Random Forest (Ensemble) | Best precision (0.7611) among all models. Ensemble approach with 100 trees provides robust predictions. Excellent for minimizing false positives when identifying potential subscribers. |
| XGBoost (Ensemble) | Best model overall with highest accuracy (92.27%), AUC (0.9549), and F1 score (0.6233). Gradient boosting handles class imbalance effectively. Best balance between precision and recall, making it the recommended choice for deployment. |
