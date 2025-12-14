FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded spaCy model and NLTK data
COPY nlp_models/spacy/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm
COPY nlp_models/nltk /root/nltk_data

ENV NLTK_DATA=/root/nltk_data

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
