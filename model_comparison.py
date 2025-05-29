import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import pickle
import re
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove numbers and special characters but keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(results):
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('combined_real_fake_news.csv')
    print(f"Dataset shape: {df.shape}")

    # Combine title and content
    print("Preprocessing text data...")
    df['text'] = df['title'] + " " + df['content']
    df['text'] = df['text'].apply(preprocess_text)

    # Split features and target
    X = df['text']
    y = df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
        'Linear SVC': LinearSVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate model
        results[name] = evaluate_model(y_test, y_pred, name)
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"F1 Score: {results[name]['f1_score']:.4f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
        
        # Plot confusion matrix
        plot_confusion_matrix(results[name]['confusion_matrix'], name, f'confusion_matrix_{name.lower().replace(" ", "_")}.png')

    # Plot model comparison
    plot_model_comparison(results)

    # Find the best model based on F1 score
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"\nBest performing model (F1 Score): {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

    # Save detailed results to a text file
    with open('model_comparison_results.txt', 'w') as f:
        for name, result in results.items():
            f.write(f"\n{name} Results:\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(result['classification_report'])
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 