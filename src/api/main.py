#this is main.py
from fastapi import FastAPI
from src.api.predict import predict_news
from src.config.logger import logger

app = FastAPI()

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Fake News Detection API"}

app.include_router(predict_news)