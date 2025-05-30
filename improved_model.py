import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class FakeNewsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract fake news specific features"""
    
    def __init__(self):
        self.viral_keywords = [
            'viral', 'tular', 'kongsi', 'share', 'viral di media sosial',
            'tular di media sosial', 'dikongsi', 'disebarkan'
        ]
        self.credible_sources = [
            'kementerian', 'jabatan', 'polis', 'bank negara', 'perdana menteri',
            'kerajaan', 'hospital', 'universiti', 'pdrm', 'sprm', 'mahkamah'
        ]
        self.fake_patterns = [
            r'kononnya', r'katanya', r'didakwa', r'dikatakan', r'dipercayai',
            r'viral di whatsapp', r'viral di facebook', r'viral di tiktok'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = np.zeros((len(X), 6))
        
        for i, text in enumerate(X):
            text = str(text).lower()
            
            # Feature 1: Count of viral keywords
            features[i, 0] = sum(1 for keyword in self.viral_keywords if keyword in text)
            
            # Feature 2: Count of credible sources mentioned
            features[i, 1] = sum(1 for source in self.credible_sources if source in text)
            
            # Feature 3: Count of fake news patterns
            features[i, 2] = sum(1 for pattern in self.fake_patterns if re.search(pattern, text))
            
            # Feature 4: Presence of denial statements
            features[i, 3] = 1 if any(word in text for word in ['menafikan', 'menolak', 'membantah', 'tidak benar']) else 0
            
            # Feature 5: Presence of verification statements
            features[i, 4] = 1 if any(word in text for word in ['sahih', 'disahkan', 'rasmi', 'pengesahan']) else 0
            
            # Feature 6: Ratio of uppercase words (shouting)
            words = text.split()
            if words:
                features[i, 5] = sum(1 for word in words if word.isupper()) / len(words)
        
        return features

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Keep case information (important for shouting detection)
    original_text = text
    
    # Convert to lowercase for other processing
    text = text.lower()
    
    # Remove URLs but keep the word 'viral' if present in URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Add back some original case information for shouting detection
    original_words = original_text.split()
    if original_words:
        uppercase_words = [word for word in original_words if word.isupper()]
        if uppercase_words:
            text += ' ' + ' '.join(uppercase_words)
    
    return text

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

    # Create feature union pipeline
    feature_union = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.95
        )),
        ('fake_news_features', FakeNewsFeatureExtractor())
    ])

    # Create the full pipeline
    pipeline = Pipeline([
        ('features', feature_union),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    # Train the model
    print("\nTraining the improved model...")
    pipeline.fit(X_train, y_train)

    # Make predictions
    print("\nEvaluating the model...")
    y_pred = pipeline.predict(X_test)

    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('improved_confusion_matrix.png')
    plt.close()

    # Save the model using pickle
    print("\nSaving the improved model...")
    with open('improved_fake_news_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved as 'improved_fake_news_model.pkl'")

    # Feature importance analysis
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        feature_importance = pipeline.named_steps['classifier'].feature_importances_
        print("\nTop important features:")
        # Note: This will need to be interpreted carefully due to the feature union

if __name__ == "__main__":
    main() 