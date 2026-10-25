import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Emotion order used consistently in the paper
# ============================================================

labels = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ============================================================
# 1. Naive Bayes confusion matrix
# ============================================================

# Original NB matrix:
# Numeric label order:
# 0 = Sadness
# 1 = Joy
# 2 = Love
# 3 = Anger
# 4 = Fear
# 5 = Surprise

nb_cm = np.array([
    [107, 6, 6, 15, 11, 6],
    [7, 78, 14, 11, 6, 12],
    [0, 1, 137, 1, 0, 0],
    [5, 2, 2, 124, 1, 1],
    [2, 0, 1, 0, 133, 1],
    [0, 0, 0, 0, 0, 144]
])


# ============================================================
# Plot Naive Bayes confusion matrix
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    nb_cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues",
    cbar=True
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Naive Bayes")

plt.tight_layout()

plt.savefig(
    "naive_bayes_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. LSTM confusion matrix
# ============================================================

# Original LSTM matrix:
#
# 0 = Sadness
# 1 = Joy
# 2 = Love
# 3 = Anger
# 4 = Fear
# 5 = Surprise

lstm_cm = np.array([
    [103, 21, 0, 15, 0, 0],
    [8, 104, 9, 13, 3, 2],
    [0, 3, 135, 1, 0, 0],
    [0, 5, 0, 134, 0, 0],
    [6, 5, 0, 1, 127, 0],
    [0, 0, 0, 0, 0, 139]
])


# ============================================================
# Plot LSTM confusion matrix
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    lstm_cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues",
    cbar=True
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - LSTM")

plt.tight_layout()

plt.savefig(
    "lstm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()