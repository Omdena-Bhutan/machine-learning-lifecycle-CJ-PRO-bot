import json
import pickle
import numpy as np

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


MODEL_DIR = "models/trained"
TEST_PKL = "data/processed/test.pkl"
OUT_JSON = "eval_metrics.json"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    test_data = load_pickle(TEST_PKL)
    test_dataset = Dataset.from_dict(test_data).shuffle(seed=42).select(range(1000))  # keep CPU-friendly

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

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

    metrics = trainer.evaluate()

    # Trainer returns keys like eval_accuracy; normalize them
    clean = {
        "eval_loss": float(metrics.get("eval_loss", 0.0)),
        "eval_accuracy": float(metrics.get("eval_accuracy", metrics.get("accuracy", 0.0))),
        "eval_f1": float(metrics.get("eval_f1", metrics.get("f1", 0.0))),
        "eval_precision": float(metrics.get("eval_precision", metrics.get("precision", 0.0))),
        "eval_recall": float(metrics.get("eval_recall", metrics.get("recall", 0.0))),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(clean, f, indent=2)

    print("[OK] Wrote", OUT_JSON)
    print(clean)


if __name__ == "__main__":
    main()
