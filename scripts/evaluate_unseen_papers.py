"""Evaluate the subfield classifier on the test split and on the part of it that has no leakage.

Every paper in the dataset has two records (an abstract and a summary) with identical labels,
and the released splits were made by record. Many test records therefore belong to papers that
also occur in the training split. This script reports the metrics

  - on all test records,
  - on test records whose paper has no record in the training split, and
  - on test records whose paper has no record in the training or the validation split.

Examples:
    python scripts/evaluate_unseen_papers.py
    python scripts/evaluate_unseen_papers.py --dataset path/to/folder_with_jsonl --model path/to/mlp_model.h5
"""
import argparse
import json

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow import keras

DEFAULT_MODEL = "Bocklitz-Lab/lit2vec-subfield-classifier-model"
DEFAULT_DATASET = "Bocklitz-Lab/lit2vec-subfield-classifier-dataset"
NUM_CLASSES = 18


def label_matrix(labels):
    y = np.zeros((len(labels), NUM_CLASSES), dtype=int)
    for i, ids in enumerate(labels):
        for j in ids:
            y[i, j] = 1
    return y


def metrics(y, p, threshold):
    pred = (p > threshold).astype(int)
    has_both = [j for j in range(NUM_CLASSES) if 0 < y[:, j].sum() < len(y)]
    f1_per_label = {j: f1_score(y[:, j], pred[:, j], zero_division=0) for j in has_both}
    return {
        "records": len(y),
        "micro_f1": f1_score(y, pred, average="micro", zero_division=0),
        "macro_f1": float(np.mean(list(f1_per_label.values()))),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "samples_f1": f1_score(y, pred, average="samples", zero_division=0),
        "micro_precision": precision_score(y, pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y, pred, average="micro", zero_division=0),
        "micro_roc_auc": roc_auc_score(y.ravel(), p.ravel()),
        "micro_pr_auc": average_precision_score(y.ravel(), p.ravel()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or path to an .h5 file")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id or a folder with the JSONL splits")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Optional path for the results as JSON")
    args = parser.parse_args()

    model_path = args.model if args.model.endswith(".h5") else hf_hub_download(args.model, "mlp_model.h5")
    model = keras.models.load_model(model_path, compile=False)

    ds = load_dataset(args.dataset)
    test = ds["test"]
    train_ids = set(ds["train"]["corpus_id"])
    val_ids = set(ds["validation"]["corpus_id"])

    x = np.array(test["embedding"], dtype=np.float32)
    y = label_matrix(test["label"])
    p = model.predict(x, batch_size=512, verbose=0)

    ids = np.array(test["corpus_id"])
    subsets = {
        "all test records": np.ones(len(ids), dtype=bool),
        "paper not in train": np.array([i not in train_ids for i in ids]),
        "paper not in train or validation": np.array([i not in train_ids and i not in val_ids for i in ids]),
    }

    results = {name: metrics(y[mask], p[mask], args.threshold) for name, mask in subsets.items()}

    header = f"{'subset':34s} {'n':>6s} {'micro F1':>9s} {'macro F1':>9s} {'weighted F1':>12s} {'micro PR-AUC':>13s}"
    print(header)
    for name, r in results.items():
        print(f"{name:34s} {r['records']:6d} {r['micro_f1']:9.4f} {r['macro_f1']:9.4f} {r['weighted_f1']:12.4f} {r['micro_pr_auc']:13.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
