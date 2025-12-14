"""
Text preprocessing pipeline - Docker robust version
"""
import re
import string
import contractions
import logging
from tqdm import tqdm
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing pipeline"""
    
    def __init__(self):
        logger.info("⏳ Initializing preprocessor...")
        
        try:
            # Try spaCy for local dev
            import spacy
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            self.use_spacy = True
            logger.info("✅ spaCy loaded (local dev)")
        except:
            # Docker/lightweight fallback (no lemmatization)
            self.use_spacy = False
            logger.info("✅ Using lightweight preprocessor (Docker)")
        
        self.stop_words = set(stopwords.words('english'))
        logger.info("✅ Preprocessing ready!")
    
    def preprocess_text(self, text):
        '''
        Complete text preprocessing pipeline:
        1. Lowercase, 2. HTML entities, 3. Emails, 4. HTML, 5. URLs, 
        6. Numbers, 7. Contractions, 8. Punctuation, 9. Whitespace, 10. Stopwords
        11. Lemmatization (spaCy only)
        '''
        # 1-9: Same for all environments
        text = text.lower()
        text = re.sub(r'#39;', "'", text)
        text = re.sub(r'&amp;', 'and', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = contractions.fix(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 10-11: Stopwords + Lemmatization
        if self.use_spacy:
            import spacy
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if token.text not in self.stop_words and len(token.text) > 2]
        else:
            # Lightweight: just remove stopwords, no lemmatization
            tokens = [word for word in text.split() 
                     if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text'):
        '''Apply preprocessing to entire dataframe with progress bar'''
        tqdm.pandas(desc='Preprocessing')
        df['text_clean'] = df[text_column].progress_apply(self.preprocess_text)
        return df


if __name__ == '__main__':
    preprocessor = TextPreprocessor()
    test_text = 'Wall St. Bears 32#8 & Claw Back Into the Black (Reuters)'
    processed = preprocessor.preprocess_text(test_text)
    print(f'Original: {test_text}')
    print(f'Processed: {processed}')
