import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
from data_preprocessing import DataPreprocessor

def train_models():
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.generate_synthetic_data(15000)
    X, y = preprocessor.preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate models
    results = {}
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        
        # Track best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
    
    # Save the best model and preprocessor using standard pickle
    with open('mortality_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'preprocessor': preprocessor,
            'feature_names': X.columns.tolist()
        }, f)
    
    # Also save preprocessor separately for easier access
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"\nBest model: {list(models.keys())[list(results.values()).index(max(results.values(), key=lambda x: x['roc_auc']))]}")
    print(f"Best ROC AUC: {best_score:.4f}")
    
    return results, best_model

if __name__ == "__main__":
    results, best_model = train_models()