from bert_model import get_bert_predictions
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# 1. Define test data
# -------------------------
test_texts = [
    "I am very happy today!",
    "I am so sad",
    "This makes me angry"
]

# Common English labels (your current labels)
true_labels = ["happy", "sad", "angry"]

# -------------------------
# 2. Map English labels -> GoEmotions labels
# -------------------------
label_map = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    # Add more mappings if needed
}

mapped_true_labels = [label_map[label] for label in true_labels]

# -------------------------
# 3. Get BERT predictions
# -------------------------
y_true, y_pred_bert = get_bert_predictions(test_texts, mapped_true_labels)

print("y_true (mapped):", y_true)
print("y_pred_bert:", y_pred_bert)

# -------------------------
# 4. Compute metrics
# -------------------------
accuracy = accuracy_score(y_true, y_pred_bert)
precision = precision_score(y_true, y_pred_bert, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred_bert, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred_bert, average='macro', zero_division=0)

print("\n=== BERT Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# -------------------------
# 5. Classification report
# -------------------------
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_bert, digits=4, zero_division=0))

# -------------------------
# 6. Confusion matrix
# -------------------------
le = LabelEncoder()
le.fit(y_true)
cm = confusion_matrix(y_true, y_pred_bert, labels=le.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("BERT Confusion Matrix")
plt.show()

# -------------------------
# 7. Metrics table for report
# -------------------------
metrics_df = pd.DataFrame({
    "Model": ["RoBERTa"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-Score": [f1]
})
print("\n=== Metrics Table ===")
print(metrics_df)

