import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the SMS dataset

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection.tsv"

data = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

data = data.dropna(subset=["label", "message"])

X = data["message"].astype(str)
y = data["label"].astype(str).str.strip().str.lower()

# Keep only the expected classes
valid = y.isin(["ham", "spam"])
X = X[valid]
y = y[valid]

print("Total messages:", len(data))
print("Class distribution:")
print(y.value_counts())

# 2. Split into training, validation and test sets

# 20% of all data is reserved for final testing
X_remaining, X_test, y_remaining, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 20% of the remaining data is used for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.20,
    random_state=42,
    stratify=y_remaining
)

print("\nTraining messages:", len(X_train))
print("Validation messages:", len(X_val))
print("Testing messages:", len(X_test))

# 3. Create and train the ML pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

# 4. Get spam probabilities on validation data

classes = list(model.named_steps["classifier"].classes_)
spam_index = classes.index("spam")

val_probabilities = model.predict_proba(X_val)[:, spam_index]

# 5. Select a threshold using validation data

# Prefer the highest spam recall while maintaining
# at least 90% spam precision, when possible.

threshold_candidates = np.arange(0.05, 0.96, 0.01)

valid_thresholds = []

for threshold in threshold_candidates:

    val_predictions = np.where(
        val_probabilities >= threshold,
        "spam",
        "ham"
    )

    precision = precision_score(
        y_val, val_predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_val, val_predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_val, val_predictions,
        pos_label="spam",
        zero_division=0
    )

    if precision >= 0.90:
        valid_thresholds.append(
            (recall, f1, threshold, precision)
        )

if valid_thresholds:
    # Highest recall first, then F1 as a tie-breaker
    best_recall, best_f1, best_threshold, best_precision = max(
        valid_thresholds,
        key=lambda item: (item[0], item[1])
    )

else:
    # If no threshold achieves 90% precision,
    # choose the threshold with the best F1 score.
    results = []

    for threshold in threshold_candidates:

        val_predictions = np.where(
            val_probabilities >= threshold,
            "spam",
            "ham"
        )

        precision = precision_score(
            y_val, val_predictions,
            pos_label="spam",
            zero_division=0
        )

        recall = recall_score(
            y_val, val_predictions,
            pos_label="spam",
            zero_division=0
        )

        f1 = f1_score(
            y_val, val_predictions,
            pos_label="spam",
            zero_division=0
        )

        results.append(
            (f1, recall, threshold, precision)
        )

    best_f1, best_recall, best_threshold, best_precision = max(
        results,
        key=lambda item: (item[0], item[1])
    )

print("\nSelected spam threshold:", round(best_threshold, 2))
print("Validation spam precision:", round(best_precision, 4))
print("Validation spam recall:", round(best_recall, 4))
print("Validation spam F1-score:", round(best_f1, 4))

# 6. Evaluate on the untouched test set

test_probabilities = model.predict_proba(X_test)[:, spam_index]

y_pred = np.where(
    test_probabilities >= best_threshold,
    "spam",
    "ham"
)

print("\nFINAL TEST RESULTS")
print("------------------")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)

print("Spam precision:", round(
    precision_score(y_test, y_pred, pos_label="spam", zero_division=0), 4
))

print("Spam recall:", round(
    recall_score(y_test, y_pred, pos_label="spam", zero_division=0), 4
))

print("Spam F1-score:", round(
    f1_score(y_test, y_pred, pos_label="spam", zero_division=0), 4
))

# 7. Display the confusion matrix

labels = ["ham", "spam"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Spam", "Spam"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Spam Detection - Tuned Threshold")
plt.tight_layout()
plt.show()

# 8. Save model AND selected threshold

MODEL_PATH = BASE_DIR / "spam_model.pkl"

saved_data = {
    "model": model,
    "threshold": float(best_threshold)
}

with open(MODEL_PATH, "wb") as file:
    pickle.dump(saved_data, file)

print("\nModel saved at:", MODEL_PATH)
print("Selected threshold saved successfully.")

# --------------------------------------------------
# SIGNIFICANCE:
# Validation data selects the threshold without using
# the final test labels.
#
# A lower threshold can detect more spam but may also
# flag more legitimate messages incorrectly.
#
# We prefer high spam recall while trying to maintain
# at least 90% spam precision on validation data.
#
# Test results show how well this choice generalizes
# to messages that were not used for training or tuning.
# --------------------------------------------------