# Bank Marketing Campaign Classification

A machine learning project implementing 6 classification models to predict term deposit subscriptions from bank marketing campaigns.

## Problem Statement

Predict whether a client will subscribe to a term deposit based on marketing campaign data (phone calls) from a Portuguese bank. This is a binary classification problem - clients either subscribe (Yes) or don't subscribe (No).

The goal is to help the bank optimize marketing strategy by identifying potential subscribers.

## Dataset

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

All models trained on 80-20 train-test split:

### Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.9147 | 0.8920 | 0.6347 | 0.5711 | 0.6012 | 0.5546 |
| KNN | 0.9053 | 0.8617 | 0.6267 | 0.3944 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest | 0.9202 | 0.9515 | 0.7611 | 0.4256 | 0.5460 | 0.5318 |
| XGBoost | 0.9227 | 0.9549 | 0.6907 | 0.5679 | 0.6233 | 0.5841 |

**Observations:**
- XGBoost performed best overall with highest accuracy (92.27%) and F1 score (0.6233)
- Random Forest had highest precision (0.7611)
- Naive Bayes had highest recall (0.6175) but lowest accuracy
- Ensemble methods outperformed single classifiers
- Dataset imbalance affects all models

### Model Analysis

**Logistic Regression:** Good baseline with 91.39% accuracy. High precision but low recall means it's conservative in predictions.

**Decision Tree:** Balanced precision-recall with F1 of 0.60. max_depth=10 prevented overfitting.

**KNN:** Lowest performer due to curse of dimensionality. Not suitable for this dataset.

**Naive Bayes:** Fast but assumes independence between features which isn't true here.

**Random Forest:** Best precision. Good for minimizing false positives.

**XGBoost:** Best model overall. Good balance and handles imbalanced data well.

## Installation

**Requirements:**
- Python 3.8+
- pip

**Steps:**

1. Clone repository
```bash
git clone https://github.com/harishperla09/ML_Term_Deposit_subscription.git
cd ML_Term_Deposit_subscription
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train all models:
```bash
python model/train_models.py
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

```
project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── bank-additional/
│       └── bank-additional-full.csv
├── model/
│   ├── train_models.py
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── evaluation_results.csv
└── utils/
    ├── data_preprocessing.py
    └── evaluation.py
```

## Application Features

1. CSV upload for test data
2. Model selection dropdown
3. Evaluation metrics display
4. Confusion matrix visualization
5. Classification report
6. Model comparison table

## Technical Details

**Preprocessing:**
- Label encoding for categorical variables
- StandardScaler for feature scaling
- 80-20 train-test split
- Random state: 42

**Hyperparameters:**
- Logistic Regression: max_iter=1000
- Decision Tree: max_depth=10
- KNN: n_neighbors=5
- Naive Bayes: default
- Random Forest: n_estimators=100, max_depth=10
- XGBoost: n_estimators=100, max_depth=6, lr=0.1

**Metrics:**
- Accuracy: Overall correctness
- AUC: Class discrimination ability
- Precision: Correct positive predictions
- Recall: Identified actual positives
- F1: Balance of precision and recall
- MCC: Balanced measure for imbalanced data

## Usage

**Train models:**
```bash
python model/train_models.py
```

**Run app:**
```bash
streamlit run app.py
```

**Using the app:**
1. Open "Test Models" tab
2. Select a model
3. Upload CSV file
4. Click "Run Prediction"

## Deployment on Streamlit Community Cloud

### Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (free at share.streamlit.io)

### Step-by-Step Deployment

**1. Prepare Your Repository**

Ensure you have:
- ✅ All source code files
- ✅ requirements.txt
- ✅ README.md
- ✅ Trained model .pkl files in model/ directory
- ✅ Dataset files in data/ directory

**2. Train Models Locally (Important!)**

```bash
python model/train_models.py
```

This creates .pkl files in the model/ directory that will be deployed.

**3. Push to GitHub**

```bash
git init
git add .
git commit -m "Bank Marketing ML Classification App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**4. Deploy on Streamlit Cloud**

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Set:
   - **Main file path:** `app.py`
   - **Python version:** 3.9+ 
5. Click "Deploy"

**5. Wait for Deployment**

Deployment takes 2-5 minutes. Your app will be live at:
`https://share.streamlit.io/YOUR_USERNAME/YOUR_REPO/main/app.py`

### Troubleshooting

**If models are missing:**
- The app will auto-train models on first load (takes 3-5 minutes)
- Or manually run: `python model/train_models.py` locally and push .pkl files

**If deployment fails:**
- Check requirements.txt has all dependencies
- Ensure Python version is 3.9 or higher
- Check Streamlit Cloud logs for errors

### Note on Model Files

The .pkl model files (~10-50 MB total) are included in the repo for easy deployment. If GitHub warns about large files:
- Use Git LFS: `git lfs track "*.pkl"`
- Or let the app auto-train on first run

## Results

Results are in `model/evaluation_results.csv` after training.

Key findings:
- Ensemble methods perform better
- Class imbalance affects metrics differently
- XGBoost is best overall

## Author

ML Assignment - M.Tech Student

## References

- UCI Machine Learning Repository
- Scikit-learn documentation
- XGBoost documentation
