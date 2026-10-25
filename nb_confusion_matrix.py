import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Load dataset
# ==========================================

data = pd.read_csv("media/balanced_data.csv")

print("Dataset shape:", data.shape)

# ==========================================
# 2. Same split as original Naive Bayes code
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 3. TF-IDF
# ==========================================

tfidf_vectorizer = TfidfVectorizer(
    max_features=10000
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# ==========================================
# 4. Train Naive Bayes
# ==========================================

nb_classifier = MultinomialNB()

nb_classifier.fit(
    X_train_tfidf,
    y_train
)

# ==========================================
# 5. Predictions
# ==========================================

predictions = nb_classifier.predict(X_test_tfidf)

# ==========================================
# 6. Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n======================================")
print("NAIVE BAYES RESULTS")
print("======================================")

print("Accuracy:", round(accuracy, 4))

# ==========================================
# 7. Classification Report
# ==========================================

label_map = {
    0: "joy",
    1: "sadness",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

class_names = [
    label_map[i]
    for i in range(6)
]

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(6)),
        target_names=class_names,
        zero_division=0
    )
)

# ==========================================
# 8. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(6))
)

print("\nConfusion Matrix:")
print(cm)

print("\nClass order:")
print(class_names)

print("\nExperiment completed.")