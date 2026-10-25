
import os
import json
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ---------------- SETTINGS ----------------
MODEL_NAME = "SamLowe/roberta-base-go_emotions"
DATA_PATH = os.path.join("media", "balanced_data.csv")
OUTPUT_DIR = os.path.join("media", "fine_tuned_roberta")
MAX_LENGTH = 128
RANDOM_STATE = 42

ID_TO_EMOTION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
LABEL_TO_ID = {v: k for k, v in ID_TO_EMOTION.items()}

# ---------------- REPRODUCIBILITY ----------------
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
if not {"text", "label"}.issubset(data.columns):
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

data = data.dropna(subset=["text", "label"]).copy()
data["text"] = data["text"].astype(str).str.strip()
data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
data = data[data["text"] != ""].copy()

if not set(data["label"].unique()).issubset(ID_TO_EMOTION):
    raise ValueError(f"Unexpected labels: {sorted(data['label'].unique())}")

print("Dataset size:", len(data))
print("Label distribution:")
print(data["label"].map(ID_TO_EMOTION).value_counts())

# Split: 80% train, 10% validation, 10% test
train_val, test_data = train_test_split(
    data, test_size=0.10, random_state=RANDOM_STATE,
    stratify=data["label"]
)
train_data, val_data = train_test_split(
    train_val, test_size=1/9, random_state=RANDOM_STATE,
    stratify=train_val["label"]
)

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

class EmotionDataset(Dataset):
    def __init__(self, frame):
        self.encodings = tokenize(frame["text"].tolist())
        self.labels = torch.tensor(frame["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = EmotionDataset(train_data)
val_dataset = EmotionDataset(val_data)
test_dataset = EmotionDataset(test_data)

# ---------------- MODEL ----------------
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=len(ID_TO_EMOTION),
    id2label=ID_TO_EMOTION,
    label2id=LABEL_TO_ID,
    problem_type="single_label_classification"
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True
)

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir="./roberta_training_checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./roberta_training_logs",
    logging_steps=25,
    report_to="none",
    seed=RANDOM_STATE
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ---------------- FINAL TEST EVALUATION ----------------
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=-1)

class_ids = list(ID_TO_EMOTION.keys())
class_names = [ID_TO_EMOTION[i] for i in class_ids]

metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred))
}
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
metrics.update({
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "test_samples": int(len(y_true)),
    "train_samples": int(len(train_data)),
    "validation_samples": int(len(val_data)),
    "random_state": RANDOM_STATE,
    "epochs": 2
})

report = classification_report(
    y_true, y_pred, labels=class_ids,
    target_names=class_names, output_dict=True, zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=class_ids)

# ---------------- SAVE ALL RESULTS ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

pd.DataFrame({
    "text": test_data["text"].tolist(),
    "true_label_id": y_true,
    "true_emotion": [ID_TO_EMOTION[int(i)] for i in y_true],
    "predicted_label_id": y_pred,
    "predicted_emotion": [ID_TO_EMOTION[int(i)] for i in y_pred]
}).to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "emotion_labels.json"), "w") as f:
    json.dump({str(k): v for k, v in ID_TO_EMOTION.items()}, f, indent=4)

print("\nFINAL TEST METRICS")
print(json.dumps(metrics, indent=4))
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true, y_pred, labels=class_ids,
    target_names=class_names, zero_division=0
))
print("\nCONFUSION MATRIX")
print(pd.DataFrame(cm, index=class_names, columns=class_names))
print("\nSaved model and results to:", OUTPUT_DIR)
