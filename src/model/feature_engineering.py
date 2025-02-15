from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(X):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    return vectorizer, vectorizer.fit_transform(X)