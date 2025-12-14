"""
FastAPI application for AG News classification
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predict import NewsClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title='AG News Classifier API',
    version='1.0.0',
    description='REST API for AG News text classification using SVM'
)

# Global classifier instance
classifier = None


@app.on_event('startup')
async def load_model():
    '''Load model on startup'''
    global classifier
    try:
        logger.info('🚀 Loading models...')
        classifier = NewsClassifier()
        logger.info('✅ Models loaded successfully!')
    except Exception as e:
        logger.error(f'❌ Error loading models: {e}')
        raise


# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, 
                     example='Wall Street stocks rally on strong tech earnings')


class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    category: str
    category_id: int
    confidence: float
    all_probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


# Endpoints
@app.get('/', response_model=Dict[str, str])
async def root():
    '''Root endpoint'''
    return {
        'message': 'AG News Classifier API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'predict_batch': '/predict/batch',
            'docs': '/docs'
        }
    }


@app.get('/health', response_model=HealthResponse)
async def health_check():
    '''Health check endpoint'''
    return {
        'status': 'healthy',
        'version': '1.0.0',
        'model_loaded': classifier is not None
    }


@app.post('/predict', response_model=PredictionResponse)
async def predict(input_data: TextInput):
    '''
    Predict news category for input text
    
    Args:
        input_data: TextInput with news text
        
    Returns:
        PredictionResponse with category and confidence
    '''
    if classifier is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        result = classifier.predict(input_data.text)
        return result
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


@app.post('/predict/batch', response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    '''
    Predict news categories for multiple texts
    
    Args:
        input_data: BatchTextInput with list of texts
        
    Returns:
        BatchPredictionResponse with predictions
    '''
    if classifier is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        results = classifier.predict_batch(input_data.texts)
        return {'predictions': results}
    except Exception as e:
        logger.error(f'Batch prediction error: {e}')
        raise HTTPException(status_code=500, detail=f'Batch prediction failed: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
