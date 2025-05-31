from sklearn.base import BaseEstimator, TransformerMixin
import re

class FakeNewsIndicatorExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer for extracting fake news indicators from text"""
    
    def __init__(self):
        # List of words that might indicate fake news in Malay
        self.indicators = [
            'belum disahkan',
            'dakwaan',
            'desas-desus',
            'didakwa',
            'dikatakan',
            'dipersoalkan',
            'khabar angin',
            'konon',
            'kononnya',
            'palsu',
            'spekulasi',
            'tidak rasmi',
            'tidak sahih',
            'tular',
            'viral'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform the input text by checking for fake news indicators"""
        if isinstance(X, str):
            X = [X]
        
        # Convert all text to lowercase for matching
        X = [str(text).lower() for text in X]
        
        # Check for each indicator in the text
        result = []
        for text in X:
            # Count how many indicators are present
            count = sum(1 for indicator in self.indicators if indicator in text)
            result.append([count])
        
        return result 