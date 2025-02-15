import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, accuracy_score
from src.data.data_loader import load_data
from src.config.logger import logger
from src.model.feature_engineering import preprocess_text


#load the dataset 
df = load_data() 
X,y = df['text'], df['label']

#convert the text data into numerical data
vectorizer , X_tfidf = preprocess_text(X)

#split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train SVM model with hyperparameter tuning
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": [0.1, 0.01, 0.001]}
svm_model = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
svm_model.fit(X_train, y_train)

# Save the trained model
with open("model/model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    
# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info("Model Accuracy: %s", accuracy)
print(f"Model Accuracy: {accuracy}")