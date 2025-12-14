# 📰 AG News Classification - MLOps on Azure

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Azure](https://img.shields.io/badge/Azure-Cloud-0078D4)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview

This project demonstrates a production-ready **End-to-End MLOps Pipeline** for an NLP classification model using the AG News dataset. The system classifies news articles into four categories: **World, Sports, Business, and Science/Tech**.

The pipeline includes model training, containerization, cloud deployment on Azure, and automated CI/CD with GitHub Actions.

---

## 🚀 Key Features

- **🤖 NLP Model**: Linear SVM with TF-IDF Vectorization (91.34% Accuracy).
- **🐳 Containerization**: Dockerized API for consistent deployment.
- **☁️ Cloud Deployment**: Deployed on Azure Container Instances (ACI) via Azure Container Registry (ACR).
- **⚡ REST API**: High-performance API built with FastAPI.
- **🔄 CI/CD**: Automated pipeline using GitHub Actions.
- **📊 Monitoring**: Integrated with Azure Application Insights.
- **🎨 Frontend**: Interactive UI built with Streamlit.

---

## � App Demo

### 🔹 Single Text Prediction
<!-- Replace the path below with your actual image path, e.g., docs/images/single_pred.png -->
![Single Prediction](<img width="688" height="664" alt="image" src="https://github.com/user-attachments/assets/6dbb8091-9be5-46e7-b28c-c5b75e6402ee" />
t)
*Enter text and get real-time classification.*


---

## ⚙️ Deployment & CI/CD Visuals

### 🔹 GitHub Actions Pipeline
<!-- Replace with screenshot of your GitHub Actions 'Success' run -->
![GitHub Actions](<img width="983" height="151" alt="image" src="https://github.com/user-attachments/assets/e9026262-e422-4499-abe9-719f1548e1b6" />
)
*Automated build and deploy pipeline triggered on push.*

### 🔹 Azure Resources
<!-- Replace with screenshot of your Azure Resource Group (ACR + ACI + App Insights) -->
![Azure Portal](<img width="1207" height="113" alt="image" src="https://github.com/user-attachments/assets/ce83231a-ae69-4e0e-bea6-347f4dff4b0f" />
)
*Deployed resources: Container Registry, Container Instance, and Application Insights.*

---

## �🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 🐍 |
| **ML Framework** | Scikit-learn, NLTK, Spacy |
| **API Framework** | FastAPI |
| **Containerization** | Docker |
| **Cloud Provider** | Microsoft Azure (ACI, ACR) |
| **CI/CD** | GitHub Actions |
| **Frontend** | Streamlit |
| **Tracking** | MLflow |

---

## 📊 Model Performance

The model was evaluated on the AG News test dataset.

| Metric | Value |
|--------|-------|
| **Model Architecture** | Linear SVM |
| **Accuracy** | **91.34%** |
| **Training Time** | ~3 seconds |
| **Vectorization** | TF-IDF (Unigrams + Bigrams) |

> *Note: Detailed evaluation metrics and confusion matrix can be found in the notebooks directory.*

---

## 📂 Project Structure

```
MLOps Azure/
├── .github/workflows/    # CI/CD Pipelines
├── config/               # Configuration files (params.yaml)
├── data/                 # Raw and processed data
├── docker/               # Dockerfile
├── src/
│   ├── api/              # FastAPI application
│   ├── data/             # Data loading scripts
│   ├── features/         # Preprocessing logic
│   ├── frontend/         # Streamlit UI
│   └── models/           # Training and prediction logic
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚡ Getting Started

### 1️⃣ Prerequisites
- Python 3.9+
- Docker Desktop
- Azure CLI

### 2️⃣ Local Installation

```bash
# Clone the repository
git clone https://github.com/YassirCher/ag-news-mlops-azure.git
cd "MLOps Azure"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Running Locally

**Train the Model:**
```bash
python src/models/train.py
```

**Run the API:**
```bash
uvicorn src.api.app:app --reload
```
*Access API docs at: http://localhost:8000/docs*

**Run the Frontend:**
```bash
streamlit run src/frontend/app_ui.py
```

---

## 🐳 Docker Support

Build and run the container locally:

```bash
# Build image
docker build -f docker/Dockerfile -t ag-news-svm-api:latest .

# Run container
docker run -p 8000:8000 ag-news-svm-api:latest
```

---

## ☁️ Azure Deployment

The project is deployed using Azure Container Instances (ACI).

### 1. Setup Resources
```powershell
# Login to Azure
az login

# Create Resource Group
az group create --name "ag-news-mlops-rg" --location "spaincentral"

# Create Container Registry
az acr create --resource-group "ag-news-mlops-rg" --name <your_acr_name> --sku Basic --admin-enabled true
```

### 2. Push to Azure Container Registry (ACR)
```powershell
# Login to ACR
az acr login --name <your_acr_name>

# Tag and Push
docker tag ag-news-svm-api:latest <your_acr_name>.azurecr.io/ag-news-svm-api:latest
docker push <your_acr_name>.azurecr.io/ag-news-svm-api:latest
```

### 3. Deploy to ACI
```powershell
az container create \
  --resource-group "ag-news-mlops-rg" \
  --name "ag-news-api-container" \
  --image <your_acr_name>.azurecr.io/ag-news-svm-api:latest \
  --dns-name-label "agnews-api" \
  --ports 8000
```

---

## 🔄 CI/CD Pipeline

The project uses **GitHub Actions** for continuous integration and deployment.

- **Trigger**: Push to `main` branch.
- **Steps**:
    1. Checkout code.
    2. Login to Azure.
    3. Build Docker image.
    4. Push to Azure Container Registry.
    5. Update Azure Container Instance.

**Secrets Required:** `AZURE_CREDENTIALS`, `REGISTRY_LOGIN_SERVER`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD`, `RESOURCE_GROUP`.

---

## 📝 License

This project is licensed under the MIT License.
