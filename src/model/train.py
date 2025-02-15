import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from src.data.data_loader import load_data
from src.config.logger import logger
from src.model.feature_engineering import preprocess_text

logger.info("Starting model training...")

# Load the processed dataset
df, dataset_path = load_data()  # Now loads processed_data.csv
if df is None or df.empty:
    logger.error("Dataset could not be loaded or is empty.")
    exit()

logger.info(f"Dataset loaded from: {dataset_path}")
logger.info(f"First 5 rows of the dataset:\n{df.head()}")  # Print first 5 rows for verification

# Extract text and labels
X, y = df['text'], df['label']

# Convert the text data into numerical features
vectorizer, X_tfidf = preprocess_text(X)
if X_tfidf is None:
    logger.error("Feature extraction failed.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

logger.info("Training data size: %d", X_train.shape[0])
logger.info("Testing data size: %d", X_test.shape[0])

# Define hyperparameter grid for SVM
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": [0.1, 0.01, 0.001]
}

# Train SVM model with hyperparameter tuning
logger.info("Starting model training with hyperparameter tuning...")
svm_model = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
svm_model.fit(X_train, y_train)

# Ensure model directory exists
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Save the trained model and vectorizer
with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(svm_model, f)
with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

logger.info("Model and vectorizer saved successfully.")

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info("Model Accuracy: %.4f", accuracy)
print(f"Model Accuracy: {accuracy:.4f}")

# Print detailed classification report
report = classification_report(y_test, y_pred)
logger.info("Classification Report:\n%s", report)
print(report)
