"""
Fake News Detection System
A comprehensive NLP-based system to detect fake news using multiple machine learning models.
Target F1-score: 0.92
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.models.logistic_model import LogisticModel
from src.models.svm_model import SVMModel
from src.models.lstm_model import LSTMModel
from src.evaluator import ModelEvaluator

def main():
    """Main function to run the fake news detection pipeline"""
    print("🚀 Starting Fake News Detection System")
    print("=" * 50)
    
    # Step 1: Load Data
    print("📊 Loading dataset...")
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Step 2: Exploratory Data Analysis
    print("🔍 Performing EDA...")
    data_loader.perform_eda(df)
    
    # Step 3: Text Preprocessing
    print("🧹 Preprocessing text data...")
    preprocessor = TextPreprocessor()
    df['cleaned_text'] = preprocessor.preprocess_text(df['text'])
    
    # Step 4: TF-IDF Vectorization
    print("🔢 Applying TF-IDF vectorization...")
    X_tfidf, y = preprocessor.apply_tfidf(df['cleaned_text'], df['label'])
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 6: Train Models
    models = {}
    evaluator = ModelEvaluator()
    
    # Logistic Regression
    print("🤖 Training Logistic Regression...")
    lr_model = LogisticModel()
    models['Logistic Regression'] = lr_model.train(X_train, y_train)
    
    # SVM
    print("🤖 Training SVM...")
    svm_model = SVMModel()
    models['SVM'] = svm_model.train(X_train, y_train)
    
    # LSTM (requires different data preparation)
    print("🤖 Training LSTM...")
    lstm_model = LSTMModel()
    models['LSTM'] = lstm_model.train(df['cleaned_text'], df['label'])
    
    # Step 7: Evaluate Models
    print("📈 Evaluating models...")
    results = evaluator.evaluate_all_models(models, X_test, y_test, df)
    
    # Step 8: Display Results
    evaluator.display_results(results)
    
    print("✅ Fake News Detection System Complete!")

if __name__ == "__main__":
    main()