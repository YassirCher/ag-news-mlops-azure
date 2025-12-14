# AG News MLOps - Azure Deployment

Production-ready text classification system for AG News dataset with MLOps best practices.

## 🎯 Features

- ✅ SVM classifier with 91.34% accuracy
- ✅ MLflow experiment tracking
- ✅ FastAPI REST endpoint
- ✅ Docker containerization
- ✅ Azure deployment (Container Instances)
- ✅ CI/CD with GitHub Actions
- ✅ Monitoring with Azure Application Insights

## 📊 Model Performance

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Linear SVM | 91.34% | 3 seconds |

## 🚀 Quick Start

\\\ash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/models/train.py

# Run API locally
uvicorn src.api.app:app --reload

# Test API
curl http://localhost:8000/health
\\\

## 📁 Project Structure

See directory tree above.

## 🔧 Configuration

Edit \config/params.yaml\ for hyperparameters.

## 📝 License

MIT
