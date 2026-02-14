# Bank Marketing Classification App
# Complete self-contained application with training and prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

# ==================== DATA PREPROCESSING ====================

def preprocess_data(data_path='data/bank-additional/bank-additional-full.csv'):
    """Preprocess the dataset and return train/test splits"""
    
    # Load data
    df = pd.read_csv(data_path, sep=';')
    st.write(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y'].map({'yes': 1, 'no': 0})
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    st.write(f"✓ Categorical features: {len(categorical_cols)}")
    st.write(f"✓ Numerical features: {len(numerical_cols)}")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    st.write(f"✓ Train set: {len(X_train)} samples")
    st.write(f"✓ Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X_encoded.columns.tolist()
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Save preprocessor
    os.makedirs('model', exist_ok=True)
    with open('model/preprocessor.pkl', 'wb') as f:
        pickle.dump({
            'label_encoders': label_encoders,
            'scaler': scaler,
            'feature_names': feature_names
        }, f)
    
    st.write("✓ Preprocessor saved")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==================== MODEL TRAINING ====================

def initialize_models():
    """Initialize all 6 models"""
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss')
    }

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all models"""
    
    models = initialize_models()
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for idx, (model_name, model) in enumerate(models.items()):
        status_text.text(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba is not None else roc_auc_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)
        
        # Save model
        filepath = os.path.join('model', model_files[model_name])
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("✓ All models trained and saved!")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('model/evaluation_results.csv', index=False)
    
    return results_df, trained_models

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_models():
    """Load all trained models from pkl files"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbors': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    return models

@st.cache_resource
def load_preprocessor():
    """Load preprocessor from file"""
    preprocessor_path = 'model/preprocessor.pkl'
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
    return None

def transform_data(X, preprocessor_dict):
    """Transform new data using the preprocessor"""
    X_transformed = X.copy()
    
    # Encode categorical variables
    label_encoders = preprocessor_dict['label_encoders']
    for col, le in label_encoders.items():
        if col in X_transformed.columns:
            X_transformed[col] = le.transform(X_transformed[col].astype(str))
    
    # Scale features
    scaler = preprocessor_dict['scaler']
    X_scaled = scaler.transform(X_transformed)
    
    # Convert back to DataFrame
    feature_names = preprocessor_dict['feature_names']
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    return X_scaled

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC
    if y_pred_proba is not None:
        try:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = roc_auc_score(y_true, y_pred)
    else:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['AUC'] = 0.0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No', 'Yes'], 
                yticklabels=['No', 'Yes'],
                ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    st.title("Bank Marketing Campaign Prediction")
    st.write("Test trained ML models for predicting term deposit subscription")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About the Project")
        st.write("""
        This project compares 6 machine learning models on the Bank Marketing dataset.
        
        Test Dataset: 4,121 records, 20 features
        Task: Binary classification (yes/no)
        """)
        
        # Download dataset button
        st.markdown("---")
        st.header("Download Test Dataset")
        
        dataset_path = 'data/bank-additional/bank-additional.csv'
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                st.download_button(
                    label="Download Test Dataset",
                    data=f,
                    file_name="bank-additional.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        st.header("Models Used")
        st.write("""
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors
        4. Naive Bayes
        5. Random Forest
        6. XGBoost
        """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Test Models", "Dataset Info"])
    
    # TAB 1: Testing
    with tab1:
        st.header("Test with Your Data")
        
        # Load models
        models = load_models()
        preprocessor = load_preprocessor()
        
        if not models:
            st.error("No trained models found in the model/ directory.")
            st.info("Models should be available after uploading to GitHub. Make sure all .pkl files are in the model/ folder.")
            return
        
        # Select model
        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys())
        )
        
        st.markdown("---")
        
        # Upload CSV
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file (test data)",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                # Read data
                test_data = pd.read_csv(uploaded_file, sep=';')
                st.success(f"File uploaded! Shape: {test_data.shape}")
                
                # Preview data
                with st.expander("Preview data"):
                    st.dataframe(test_data.head(10))
                
                # Check for target column
                has_target = 'y' in test_data.columns
                
                if has_target:
                    X_test = test_data.drop('y', axis=1)
                    y_test = test_data['y'].map({'yes': 1, 'no': 0})
                else:
                    X_test = test_data
                    y_test = None
                
                # Check preprocessor
                if preprocessor is None:
                    st.error("Preprocessor not found! Train models first.")
                    st.stop()
                
                # Predictions
                model = models[selected_model]
                
                if st.button("Run Prediction", type="primary"):
                    with st.spinner(f"Running {selected_model}..."):
                        # Preprocess data
                        X_test_processed = transform_data(X_test, preprocessor)
                        
                        y_pred = model.predict(X_test_processed)
                        
                        # Get probabilities
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test_processed)
                        else:
                            y_pred_proba = None
                        
                        # Show predictions
                        st.subheader("Predictions")
                        pred_df = pd.DataFrame({
                            'Prediction': ['Yes' if p == 1 else 'No' for p in y_pred],
                            'Confidence': y_pred_proba[:, 1] if y_pred_proba is not None else y_pred
                        })
                        st.dataframe(pred_df.head(20), use_container_width=True)
                        
                        # Evaluation if target exists
                        if has_target:
                            st.markdown("---")
                            
                            # Metrics
                            st.subheader("Evaluation Metrics")
                            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                            
                            # Display in columns
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                st.metric("AUC", f"{metrics['AUC']:.4f}")
                            with col2:
                                st.metric("Precision", f"{metrics['Precision']:.4f}")
                                st.metric("Recall", f"{metrics['Recall']:.4f}")
                            with col3:
                                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                                st.metric("MCC", f"{metrics['MCC']:.4f}")
                            
                            st.markdown("---")
                            
                            # Confusion Matrix
                            st.subheader("Confusion Matrix")
                            fig = plot_confusion_matrix(y_test, y_pred, selected_model)
                            st.pyplot(fig)
                            
                            # Classification Report
                            st.subheader("Classification Report")
                            report = classification_report(y_test, y_pred, 
                                                          target_names=['No', 'Yes'])
                            st.text(report)
                            
                            # Model Comparison Section
                            st.markdown("---")
                            st.header("Compare All Models")
                            
                            if st.button("Run All Models Comparison", type="primary"):
                                with st.spinner("Evaluating all models..."):
                                    comparison_results = []
                                    
                                    for model_name, model in models.items():
                                        # Make predictions
                                        y_pred_compare = model.predict(X_test_processed)
                                        
                                        # Get probabilities
                                        if hasattr(model, 'predict_proba'):
                                            y_pred_proba_compare = model.predict_proba(X_test_processed)
                                        else:
                                            y_pred_proba_compare = None
                                        
                                        # Calculate metrics
                                        metrics_compare = calculate_metrics(y_test, y_pred_compare, y_pred_proba_compare)
                                        
                                        comparison_results.append({
                                            'Model': model_name,
                                            'Accuracy': metrics_compare['Accuracy'],
                                            'AUC': metrics_compare['AUC'],
                                            'Precision': metrics_compare['Precision'],
                                            'Recall': metrics_compare['Recall'],
                                            'F1': metrics_compare['F1 Score'],
                                            'MCC': metrics_compare['MCC']
                                        })
                                    
                                    # Create comparison dataframe
                                    comparison_df = pd.DataFrame(comparison_results)
                                    
                                    st.subheader("Model Comparison Results")
                                    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']), 
                                                use_container_width=True)
                                    
                                    # Plot metrics comparison
                                    st.subheader("Metric Comparison Visualization")
                                    
                                    metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
                                    
                                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                                    axes = axes.ravel()
                                    
                                    for idx, metric in enumerate(metrics_to_plot):
                                        axes[idx].barh(comparison_df['Model'], comparison_df[metric], color='steelblue')
                                        axes[idx].set_xlabel(metric, fontsize=10)
                                        axes[idx].set_xlim(0, 1)
                                        axes[idx].grid(axis='x', alpha=0.3)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure CSV has the correct features")
    
    # TAB 2: Dataset Info
    with tab2:
        st.header("About the Dataset")
        
        st.write("""
        ### Bank Marketing Dataset (Test Set)
        
        This dataset is from the UCI Machine Learning Repository. It contains data 
        from direct marketing campaigns of a Portuguese bank.
        
        **Source:** UCI ML Repository
        
        **Instances:** 4,121 (10% sample)
        
        **Features:** 20
        
        **Target:** Client subscription to term deposit (yes/no)
        
        ### Feature Categories
        
        **Client Information:**
        - Age, Job, Marital status, Education
        - Default status, Housing loan, Personal loan
        
        **Campaign Details:**
        - Contact type, Month, Day of week
        - Call duration, Number of contacts
        - Days since last contact
        - Previous campaign outcome
        
        **Economic Indicators:**
        - Employment variation rate
        - Consumer price index
        - Consumer confidence index
        - Euribor rate
        - Number of employees
        
        ### Reference
        
        S. Moro, P. Cortez and P. Rita. 
        A Data-Driven Approach to Predict the Success of Bank Telemarketing. 
        Decision Support Systems, Elsevier, 62:22-31, June 2014
        """)

if __name__ == "__main__":
    main()
