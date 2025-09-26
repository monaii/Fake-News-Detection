"""
Simplified Fake News Detection System
Focus on traditional ML models first (Logistic Regression and SVM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.models.logistic_model import LogisticModel
from src.models.svm_model import SVMModel

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
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Step 6: Train Models
    models = {}
    results = {}
    
    # Logistic Regression
    print("\n🤖 Training Logistic Regression...")
    lr_model = LogisticModel()
    models['Logistic Regression'] = lr_model.train(X_train, y_train, tune_hyperparameters=False)
    
    # Evaluate Logistic Regression
    lr_results = lr_model.evaluate(X_test, y_test)
    results['Logistic Regression'] = lr_results
    print(f"✅ Logistic Regression F1-Score: {lr_results['f1_score']:.4f}")
    
    # SVM
    print("\n🤖 Training SVM...")
    svm_model = SVMModel()
    models['SVM'] = svm_model.train(X_train, y_train, tune_hyperparameters=False)
    
    # Evaluate SVM
    svm_results = svm_model.evaluate(X_test, y_test)
    results['SVM'] = svm_results
    print(f"✅ SVM F1-Score: {svm_results['f1_score']:.4f}")
    
    # Step 7: Display Results
    print("\n" + "="*60)
    print("🎯 FAKE NEWS DETECTION - RESULTS SUMMARY")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        predictions = result['predictions']
        accuracy = accuracy_score(y_test, predictions)
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1-Score': result['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print(comparison_df.round(4).to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    
    # Check if target F1-score is achieved
    target_f1 = 0.92
    if best_model['F1-Score'] >= target_f1:
        print(f"\n✅ TARGET F1-SCORE ({target_f1}) ACHIEVED!")
    else:
        print(f"\n❌ Target F1-score ({target_f1}) not achieved.")
        print(f"   Gap: {target_f1 - best_model['F1-Score']:.4f}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(results, y_test, comparison_df)
    
    # Display detailed classification reports
    print("\n📋 DETAILED CLASSIFICATION REPORTS:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print("-" * 30)
        print(result['classification_report'])
    
    print("\n✅ Fake News Detection System Complete!")

def create_visualizations(results, y_test, comparison_df):
    """Create visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model comparison - F1 Score
    axes[0,0].bar(comparison_df['Model'], comparison_df['F1-Score'], 
                  color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0,0].set_title('F1-Score Comparison', fontweight='bold')
    axes[0,0].set_ylabel('F1-Score')
    axes[0,0].set_ylim(0, 1)
    axes[0,0].axhline(y=0.92, color='red', linestyle='--', alpha=0.7, label='Target F1=0.92')
    axes[0,0].legend()
    
    # Add value labels on bars
    for i, v in enumerate(comparison_df['F1-Score']):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Model comparison - Accuracy
    axes[0,1].bar(comparison_df['Model'], comparison_df['Accuracy'], 
                  color=['lightgreen', 'gold'], alpha=0.7)
    axes[0,1].set_title('Accuracy Comparison', fontweight='bold')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(comparison_df['Accuracy']):
        axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Confusion matrices
    model_names = list(results.keys())
    
    # Logistic Regression confusion matrix
    lr_predictions = results['Logistic Regression']['predictions']
    lr_cm = confusion_matrix(y_test, lr_predictions)
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
    axes[1,0].set_title('Logistic Regression\nConfusion Matrix')
    axes[1,0].set_xlabel('Predicted')
    axes[1,0].set_ylabel('Actual')
    
    # SVM confusion matrix
    svm_predictions = results['SVM']['predictions']
    svm_cm = confusion_matrix(y_test, svm_predictions)
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1,1])
    axes[1,1].set_title('SVM\nConfusion Matrix')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('results/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Evaluation plots saved to results/model_evaluation.png")

if __name__ == "__main__":
    main()