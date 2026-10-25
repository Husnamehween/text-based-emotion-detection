# bert_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ✅ Open-access RoBERTa model trained on GoEmotions (28 emotions)
MODEL = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        label = model.config.id2label[predicted_class]
    return label

print(predict_emotion("I am very happy today!"))

# ----------------------------
# New function for multiple texts (batch)
# ----------------------------
def get_bert_predictions(test_texts, true_labels=None):
    """
    Input:
        test_texts: list of strings
        true_labels: optional list of actual labels (for evaluation)
    Returns:
        y_true: list of true labels (if provided)
        y_pred: list of predicted labels
    """
    y_pred = []
    for text in test_texts:
        label = predict_emotion(text)
        y_pred.append(label)
    y_true = true_labels if true_labels is not None else []
    return y_true, y_pred

# ----------------------------
# Optional test
# ----------------------------
if __name__ == "__main__":
    test_texts = ["I am very happy today!", "I am so sad", "This makes me angry"]
    y_true = ["happy", "sad", "angry"]
    y_true, y_pred = get_bert_predictions(test_texts, y_true)
    print("Predictions:", y_pred)
