"""
Prediction module for inference
"""
import joblib
import numpy as np
from pathlib import Path
import logging
from src.features.preprocessor import TextPreprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsClassifier:
    """AG News classifier wrapper"""
    
    def __init__(
        self,
        model_path: str = "models/trained/svm_model.pkl",
        tfidf_path: str = "models/trained/tfidf_vectorizer.pkl",
    ):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained model
            tfidf_path: Path to TF-IDF vectorizer
        """
        logger.info("📥 Loading models...")
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        # Recreate preprocessor instead of unpickling (avoids WindowsPath issue)
        self.preprocessor = TextPreprocessor()
        
        self.label_mapping = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Science/Tech",
        }
        
        logger.info("✅ Models loaded successfully!")
    
    def predict(self, text: str):
        """Predict category for input text"""
        # Preprocess
        text_clean = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vector = self.tfidf.transform([text_clean])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        
        # Get decision function scores for confidence
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(text_vector)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0
            probabilities = [0.0] * 4
        
        return {
            "category": self.label_mapping[prediction],
            "category_id": int(prediction),
            "confidence": round(confidence, 4),
            "all_probabilities": {
                self.label_mapping[i]: round(float(probabilities[i]), 4)
                for i in range(4)
            },
        }
    
    def predict_batch(self, texts: list):
        """Predict for multiple texts"""
        return [self.predict(text) for text in texts]


if __name__ == '__main__':
    # Test prediction
    classifier = NewsClassifier()
    
    test_text = 'Wall Street rises as investors cheer positive earnings reports'
    result = classifier.predict(test_text)
    
    print(f'Text: {test_text}')
    print(f'Prediction: {result}')
