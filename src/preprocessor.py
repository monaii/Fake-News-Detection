"""
Text Preprocessing Module
Handles comprehensive text preprocessing and TF-IDF vectorization
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        
    def clean_text(self, text):
        """
        Clean text by removing special characters, URLs, etc.
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        """
        if not text:
            return []
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text_series, use_stemming=True):
        """
        Complete preprocessing pipeline for text data
        """
        print("🧹 Starting text preprocessing...")
        
        processed_texts = []
        
        for i, text in enumerate(text_series):
            if i % 5 == 0:  # Progress indicator
                print(f"Processing text {i+1}/{len(text_series)}")
            
            # Step 1: Clean text
            cleaned = self.clean_text(text)
            
            # Step 2: Tokenize
            tokens = self.tokenize_text(cleaned)
            
            # Step 3: Remove stopwords
            tokens = self.remove_stopwords(tokens)
            
            # Step 4: Apply stemming or lemmatization
            if use_stemming:
                tokens = self.stem_tokens(tokens)
            else:
                tokens = self.lemmatize_tokens(tokens)
            
            # Step 5: Join tokens back to text
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        
        print("✅ Text preprocessing completed!")
        return processed_texts
    
    def apply_tfidf(self, texts, labels, max_features=5000, ngram_range=(1, 2)):
        """
        Apply TF-IDF vectorization to the text data
        
        Args:
            texts: List or Series of text data
            labels: Corresponding labels
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to extract
            
        Returns:
            X_tfidf: TF-IDF transformed features
            y: Labels
        """
        print("🔢 Applying TF-IDF vectorization...")
        
        # Initialize TF-IDF vectorizer with improved parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,  # Include bigrams for better context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens with 2+ chars
        )
        
        # Fit and transform the text data
        X_tfidf = self.tfidf_vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        print(f"✅ TF-IDF completed! Shape: {X_tfidf.shape}")
        print(f"📊 Features: {X_tfidf.shape[1]} unique terms")
        
        return X_tfidf, y
    
    def get_feature_names(self):
        """
        Get feature names from TF-IDF vectorizer
        """
        if self.tfidf_vectorizer:
            return self.tfidf_vectorizer.get_feature_names_out()
        return None
    
    def transform_new_text(self, text_series):
        """
        Transform new text using fitted TF-IDF vectorizer
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted yet. Call apply_tfidf first.")
        
        # Preprocess the new text
        processed_texts = self.preprocess_text(text_series)
        
        # Transform using fitted vectorizer
        return self.tfidf_vectorizer.transform(processed_texts)
    
    def get_top_features_by_class(self, X_tfidf, y, top_n=10):
        """
        Get top TF-IDF features for each class
        """
        feature_names = self.get_feature_names()
        if feature_names is None:
            return None
        
        # Convert sparse matrix to dense for easier manipulation
        X_dense = X_tfidf.toarray()
        
        results = {}
        for class_label in np.unique(y):
            # Get mean TF-IDF scores for this class
            class_mask = (y == class_label)
            mean_scores = np.mean(X_dense[class_mask], axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_scores)[-top_n:][::-1]
            top_features = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            class_name = "Real" if class_label == 0 else "Fake"
            results[class_name] = top_features
        
        return results