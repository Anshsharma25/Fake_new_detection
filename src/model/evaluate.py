from sklearn.metrics import classification_report
from src.data.data_loader import load_data
from src.model.feature_engineering import preprocess_text
import pickle
from src.config.logger import logger

logger.info("Starting model evaluation...")
df = load_data()
X, y = df["text"], df["label"]
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
X_tfidf = vectorizer.transform(X)
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
y_pred = model.predict(X_tfidf)
logger.info("Evaluation report:\n%s", classification_report(y, y_pred))
print("Evaluation report:\n", classification_report(y, y_pred))