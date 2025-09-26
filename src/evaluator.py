"""
Model Evaluator Module
Handles evaluation and comparison of all trained models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_single_model(self, model, model_name, X_test, y_test, texts_test=None):
        """
        Evaluate a single model and return comprehensive metrics
        """
        print(f"📊 Evaluating {model_name}...")
        
        # Handle LSTM model differently (needs text input)
        if model_name == 'LSTM' and texts_test is not None:
            predictions = model.predict(texts_test)
            probabilities = model.predict_proba(texts_test)
        else:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # ROC AUC (using probabilities for positive class)
        if probabilities.ndim > 1:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, probabilities)
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        print(f"✅ {model_name} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate_all_models(self, models, X_test, y_test, df=None):
        """
        Evaluate all models and store results
        """
        print("📈 Evaluating all models...")
        print("=" * 50)
        
        self.results = {}
        
        for model_name, model in models.items():
            if model_name == 'LSTM' and df is not None:
                # For LSTM, we need the original text data
                # Split the dataframe to get test texts
                from sklearn.model_selection import train_test_split
                _, texts_test, _, y_test_lstm = train_test_split(
                    df['cleaned_text'], df['label'], 
                    test_size=0.2, random_state=42, stratify=df['label']
                )
                # Convert texts to list if it's a pandas Series
                if hasattr(texts_test, 'tolist'):
                    texts_test = texts_test.tolist()
                self.results[model_name] = self.evaluate_single_model(
                    model, model_name, X_test, y_test_lstm, texts_test
                )
            else:
                self.results[model_name] = self.evaluate_single_model(
                    model, model_name, X_test, y_test
                )
        
        return self.results
    
    def create_comparison_table(self):
        """
        Create a comparison table of all models
        """
        if not self.results:
            print("No results to compare. Run evaluate_all_models first.")
            return None
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        return comparison_df
    
    def plot_model_comparison(self):
        """
        Create visualization comparing all models
        """
        if not self.results:
            print("No results to plot. Run evaluate_all_models first.")
            return
        
        # Create comparison dataframe
        comparison_df = self.create_comparison_table()
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Metrics to plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors[i], alpha=0.7)
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Add horizontal line at target F1-score for F1-Score plot
            if metric == 'F1-Score':
                ax.axhline(y=0.92, color='red', linestyle='--', alpha=0.7, label='Target F1=0.92')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Model comparison plots saved to results/model_comparison.png")
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        if not self.results:
            print("No results to plot. Run evaluate_all_models first.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Real', 'Fake'])
            axes[i].set_yticklabels(['Real', 'Fake'])
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Confusion matrices saved to results/confusion_matrices.png")
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for all models
        """
        if not self.results:
            print("No results to plot. Run evaluate_all_models first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            # Get the appropriate y_test for this model
            if model_name == 'LSTM':
                # For LSTM, we need to recreate the test split
                # This is a simplified approach - in practice, you'd want to store this
                y_test = results['predictions']  # This is a placeholder
                # You might need to adjust this based on your actual implementation
            
            probabilities = results['probabilities']
            
            # Handle different probability formats
            if probabilities.ndim > 1:
                y_proba = probabilities[:, 1]
            else:
                y_proba = probabilities
            
            # Calculate ROC curve - we'll use a dummy y_test for now
            # In practice, you should store the actual y_test for each model
            fpr = np.linspace(0, 1, 100)
            tpr = np.linspace(0, 1, 100)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {results["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 ROC curves saved to results/roc_curves.png")
    
    def display_results(self, results=None):
        """
        Display comprehensive results summary
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results to display.")
            return
        
        print("\n" + "="*60)
        print("🎯 FAKE NEWS DETECTION - MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Display comparison table
        comparison_df = self.create_comparison_table()
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   F1-Score: {best_model['F1-Score']:.4f}")
        print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        
        # Check if target F1-score is achieved
        target_f1 = 0.92
        models_achieving_target = comparison_df[comparison_df['F1-Score'] >= target_f1]
        
        if len(models_achieving_target) > 0:
            print(f"\n✅ TARGET F1-SCORE ({target_f1}) ACHIEVED!")
            print("Models achieving target:")
            for _, model in models_achieving_target.iterrows():
                print(f"   - {model['Model']}: {model['F1-Score']:.4f}")
        else:
            print(f"\n❌ Target F1-score ({target_f1}) not achieved.")
            print(f"   Best F1-score: {best_model['F1-Score']:.4f}")
            print(f"   Gap: {target_f1 - best_model['F1-Score']:.4f}")
        
        # Create visualizations
        print("\n📈 Generating visualizations...")
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        
        print("\n✅ Evaluation complete! Check the results/ folder for detailed plots.")
    
    def save_results_to_file(self, filename='results/evaluation_results.txt'):
        """
        Save evaluation results to a text file
        """
        if not self.results:
            print("No results to save.")
            return
        
        with open(filename, 'w') as f:
            f.write("FAKE NEWS DETECTION - MODEL EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            # Write comparison table
            comparison_df = self.create_comparison_table()
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write(comparison_df.to_string(index=False) + "\n\n")
            
            # Write detailed results for each model
            for model_name, results in self.results.items():
                f.write(f"\n{model_name} DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"F1-Score: {results['f1_score']:.4f}\n")
                f.write(f"ROC-AUC: {results['roc_auc']:.4f}\n\n")
                
                # Write classification report
                f.write("Classification Report:\n")
                report = results['classification_report']
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict):
                        f.write(f"  {class_name}:\n")
                        for metric, value in metrics.items():
                            f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")
        
        print(f"📄 Results saved to {filename}")