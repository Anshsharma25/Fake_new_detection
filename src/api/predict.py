from fastapi import APIRouter
import pickle
from src.api.models import NewsInput
from src.config.logger import logger

router = APIRouter()

@router.post("/predict")
def predict_news(data: NewsInput):
    logger.info("Loading model and vectorizer...")
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    logger.info("Transforming input text...")
    text_tfidf = vectorizer.transform([data.text])
    prediction = model.predict(text_tfidf)[0]
    logger.info(f"Prediction: {'Fake News' if prediction == 1 else 'Real News'}")
    return {"prediction": "Fake News" if prediction == 1 else "Real News"}