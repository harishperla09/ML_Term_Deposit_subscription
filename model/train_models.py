# Model Training Script
# Trains 6 classification models on Bank Marketing dataset

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
sys.path.append('.')
from utils.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator

class ModelTrainer:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.evaluator = ModelEvaluator()
        
    def initialize_models(self):
        # Initialize all models
        print("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
    
    def train_all_models(self, X_train, y_train):
        # Train all models
        print("\n" + "="*60)
        print("Training Models")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
                print(f"Trained {model_name}")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        print(f"\nTrained {len(self.trained_models)}/{len(self.models)} models")
    
    def evaluate_all_models(self, X_test, y_test):
        # Evaluate models
        print("\n" + "="*60)
        print("Evaluating Models")
        print("="*60)
        
        for model_name, model in self.trained_models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                else:
                    y_pred_proba = None
                
                # Evaluate model
                self.evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_proba)
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        # Print and save results
        self.evaluator.print_comparison_table()
        
        results_df = self.evaluator.save_results('model/evaluation_results.csv')
        return results_df
    
    def save_models(self, output_dir='model'):
        # Save trained models
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Saving Models")
        print("="*60)
        
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbors': 'knn.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for model_name, filename in model_files.items():
            if model_name in self.trained_models:
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(self.trained_models[model_name], f)
                print(f"Saved {model_name}")
        
        print(f"\nAll models saved to {output_dir}/")
    
    def train_evaluate_save(self, X_train, X_test, y_train, y_test):
        # Main pipeline
        self.initialize_models()
        self.train_all_models(X_train, y_train)
        results = self.evaluate_all_models(X_test, y_test)
        self.save_models()
        return results


def main():
    print("="*60)
    print("Bank Marketing Classification - Training")
    print("="*60)
    
    # Preprocess data
    print("\nStep 1: Preprocessing data")
    print("-" * 60)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
    # Train models
    print("\nStep 2: Training models")
    print("-" * 60)
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_evaluate_save(X_train, X_test, y_train, y_test)
    
    # Done
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"\nFiles saved:")
    print("  - model/preprocessor.pkl")
    print("  - model/*.pkl (models)")
    print("  - model/evaluation_results.csv")
    print(f"\nRun: streamlit run app.py")
    

if __name__ == "__main__":
    main()
