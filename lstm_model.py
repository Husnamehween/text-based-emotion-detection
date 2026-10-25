# ============================================================
# LSTM Emotion Classification
# Dataset: balanced_data.csv
# ============================================================

import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. Load dataset
# ============================================================

DATA_PATH = os.path.join("media", "balanced_data.csv")

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 3. Original Naive Bayes label mapping
# ============================================================

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

data["emotion"] = data["label"].map(label_map)


# ============================================================
# 4. Check labels
# ============================================================

print("\nClass distribution:")
print(data["emotion"].value_counts())

print("\nNumeric labels:")
print(sorted(data["label"].unique()))


# ============================================================
# 5. Text cleaning
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


data["clean_text"] = data["text"].apply(clean_text)


# ============================================================
# 6. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. Tokenization
# ============================================================

MAX_WORDS = 15000
MAX_LENGTH = 80

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)


X_train_pad = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# ============================================================
# 8. Convert labels to NumPy arrays
# ============================================================

y_train = np.array(y_train).astype("int32")
y_test = np.array(y_test).astype("int32")


# ============================================================
# 9. Build LSTM
# ============================================================

model = Sequential([

    Embedding(
    input_dim=MAX_WORDS,
    output_dim=128,
    input_length=MAX_LENGTH,
    mask_zero=True
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(
        64
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        6,
        activation="softmax"
    )
])


# ============================================================
# 10. Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. Model summary
# ============================================================

print("\nLSTM Model Architecture:\n")

model.summary()


# ============================================================
# 12. Train
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting LSTM training...\n")

history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.10,
    epochs=15,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)
print("\nTraining accuracy:")
print(history.history["accuracy"])

print("\nValidation accuracy:")
print(history.history["val_accuracy"])

print("\nTraining loss:")
print(history.history["loss"])

print("\nValidation loss:")
print(history.history["val_loss"])


print("\nNon-zero tokens in first 10 training samples:")
print(np.count_nonzero(X_train_pad[:10], axis=1))

print("\nFirst 5 cleaned training texts:")
print(X_train.head().tolist())

# 
#============================================================
# 13. Predictions
# ============================================================

probabilities = model.predict(
    X_test_pad,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)
print("\nUnique true labels:", np.unique(y_test, return_counts=True))
print("Unique predicted labels:", np.unique(y_pred, return_counts=True))

# ============================================================
# 14. Overall metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n==========================================")
print("LSTM TEST SET RESULTS")
print("==========================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))


# ============================================================
# 15. Classification Report
# ============================================================

class_names = [
    label_map[i]
    for i in range(6)
]

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(6)),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 16. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(6))
)

print("\nConfusion Matrix:\n")
print(cm)


# ============================================================
# 17. Save model
# ============================================================

model.save(
    "lstm_emotion_model.h5"
)

print("\nLSTM model saved as:")
print("lstm_emotion_model.h5")

print("\nExperiment completed successfully.")