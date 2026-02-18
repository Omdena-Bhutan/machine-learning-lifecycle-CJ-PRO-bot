import json
import pickle
import numpy as np

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

MODEL_DIR = "models/trained"
TEST_PKL = "data/processed/test.pkl"

OUT_METRICS_JSON = "eval_metrics.json"
OUT_CM_JSON = "confusion_matrix.json"
OUT_CM_CSV = "confusion_matrix.csv"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(eval_pred):
    """
    HuggingFace Trainer calls this during evaluation.
    It returns metrics that Trainer will prefix with "eval_".
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main():
    # 1) Load test dataset (CPU-friendly subset)
    test_data = load_pickle(TEST_PKL)
    test_dataset = Dataset.from_dict(test_data).shuffle(seed=42).select(range(1000))

    # 2) Load tokenizer + trained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # 3) Build Trainer for evaluation
    args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=8,
        report_to=[],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 4) Standard evaluation metrics (loss + compute_metrics)
    metrics = trainer.evaluate()

    clean = {
        "eval_loss": float(metrics.get("eval_loss", 0.0)),
        "eval_accuracy": float(metrics.get("eval_accuracy", metrics.get("accuracy", 0.0))),
        "eval_f1": float(metrics.get("eval_f1", metrics.get("f1", 0.0))),
        "eval_precision": float(metrics.get("eval_precision", metrics.get("precision", 0.0))),
        "eval_recall": float(metrics.get("eval_recall", metrics.get("recall", 0.0))),
    }

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(clean, f, indent=2)

    # 5) Confusion matrix (TN, FP, FN, TP)
    # Get predictions from the model on the same eval set
    pred_output = trainer.predict(test_dataset)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=-1)

    # labels=[0,1] ensures the order is consistent:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_dict = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    with open(OUT_CM_JSON, "w") as f:
        json.dump(cm_dict, f, indent=2)

    # Optional: also save a simple CSV for easy viewing
    with open(OUT_CM_CSV, "w") as f:
        f.write(" ,pred_0,pred_1\n")
        f.write(f"true_0,{cm[0,0]},{cm[0,1]}\n")
        f.write(f"true_1,{cm[1,0]},{cm[1,1]}\n")

    print("[OK] Wrote", OUT_METRICS_JSON, "and", OUT_CM_JSON, "and", OUT_CM_CSV)
    print("Metrics:", clean)
    print("Confusion Matrix (tn, fp, fn, tp):", cm_dict)


if __name__ == "__main__":
    main()
