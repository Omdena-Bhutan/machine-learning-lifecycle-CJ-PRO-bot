import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import json
import pickle
import yaml

import mlflow
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    # 0) Load hyperparameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    model_name = params["model_name"]
    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    lr = float(params["learning_rate"])
    weight_decay = float(params["weight_decay"])
    seed = int(params["seed"])
    train_subset = int(params["train_subset"])
    eval_subset = int(params["eval_subset"])

    # 1) Load processed data
    train_data = load_pickle("data/processed/train.pkl")
    test_data = load_pickle("data/processed/test.pkl")

    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    # 2) Subset for CPU speed (but shuffled so labels are mixed)
    train_dataset = train_dataset.shuffle(seed=seed).select(range(train_subset))
    test_dataset = test_dataset.shuffle(seed=seed).select(range(eval_subset))

    # 3) Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Training args
    training_args = TrainingArguments(
        output_dir="models/trained",
        eval_strategy="epoch",          # for your transformers version
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        load_best_model_at_end=False,
        report_to=[],
        dataloader_pin_memory=False,
    )

    # 5) MLflow run (log params so you can compare experiments)
    mlflow.start_run()
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("seed", seed)
    mlflow.log_param("train_subset", train_subset)
    mlflow.log_param("eval_subset", eval_subset)

    # 6) Train + evaluate
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    # 7) Save model + tokenizer locally (DVC will track this folder)
    os.makedirs("models/trained", exist_ok=True)
    model.save_pretrained("models/trained")
    tokenizer.save_pretrained("models/trained")

    # 8) Save metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    mlflow.end_run()

    print("\n[OK] Training completed")
    print("Final Metrics:", metrics)


if __name__ == "__main__":
    main()
