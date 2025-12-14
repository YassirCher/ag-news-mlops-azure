"""
Model training script with MLflow tracking - Using your exact training logic
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import logging

from src.data.data_loader import load_ag_news, validate_data
from src.features.preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path='config/params.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_pipeline():
    """Complete training pipeline"""
    
    logger.info('='*70)
    logger.info('🚀 AG NEWS MLOps TRAINING PIPELINE')
    logger.info('='*70)
    
    # Load config
    config = load_config()
    
    # Setup MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Load data
    logger.info('\n📥 Loading data...')
    train_df, test_df = load_ag_news(
        config['data']['train_path'],
        config['data']['test_path']
    )
    
    # Validate
    validate_data(train_df)
    validate_data(test_df)
    
    # Preprocess
    logger.info('\n🔧 Preprocessing text...')
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    test_df = preprocessor.preprocess_dataframe(test_df)
    
    # Train/val split
    logger.info('\n📊 Creating train/validation split...')
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['text_clean'],
        train_df['label'],
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state'],
        stratify=train_df['label']
    )
    
    X_test = test_df['text_clean']
    y_test = test_df['label']
    
    logger.info(f'Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')
    
    # TF-IDF vectorization
    logger.info('\n📊 Applying TF-IDF vectorization...')
    tfidf = TfidfVectorizer(
        max_features=config['tfidf']['max_features'],
        min_df=config['tfidf']['min_df'],
        max_df=config['tfidf']['max_df'],
        ngram_range=tuple(config['tfidf']['ngram_range']),
        sublinear_tf=config['tfidf']['sublinear_tf']
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    logger.info(f'TF-IDF shape: {X_train_tfidf.shape}')
    
    # Train models
    logger.info('\n🚀 Training baseline models...')
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=4),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f'\n📌 Training: {name}')
        
        with mlflow.start_run(run_name=name):
            # Train
            start_time = time.time()
            model.fit(X_train_tfidf, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_val_pred = model.predict(X_val_tfidf)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            y_test_pred = model.predict(X_test_tfidf)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Log to MLflow
            mlflow.log_param('model_type', name)
            mlflow.log_param('tfidf_max_features', config['tfidf']['max_features'])
            mlflow.log_metric('train_time', train_time)
            mlflow.log_metric('val_accuracy', val_accuracy)
            mlflow.log_metric('test_accuracy', test_accuracy)
            
            results[name] = {
                'model': model,
                'train_time': train_time,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy
            }
            
            logger.info(f'⏱️  Training time: {train_time:.2f}s')
            logger.info(f'📊 Val Acc: {val_accuracy*100:.2f}% | Test Acc: {test_accuracy*100:.2f}%')
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]['model']
    best_test_acc = results[best_model_name]['test_accuracy']
    
    logger.info(f'\n🏆 BEST MODEL: {best_model_name}')
    logger.info(f'   Test Accuracy: {best_test_acc*100:.2f}%')
    
    # Save best model
    logger.info('\n💾 Saving models...')
    Path('models/trained').mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, 'models/trained/svm_model.pkl')
    joblib.dump(tfidf, 'models/trained/tfidf_vectorizer.pkl')
    
    logger.info('✅ Models saved to models/trained/')
    
    # Final evaluation report
    y_test_pred_final = best_model.predict(X_test_tfidf)
    logger.info('\n📋 FINAL CLASSIFICATION REPORT:')
    logger.info('\n' + classification_report(
        y_test, 
        y_test_pred_final,
        target_names=['World', 'Sports', 'Business', 'Science/Tech']
    ))
    
    logger.info('\n✅ TRAINING COMPLETE!')
    
    return best_model, tfidf, preprocessor


if __name__ == '__main__':
    train_pipeline()
