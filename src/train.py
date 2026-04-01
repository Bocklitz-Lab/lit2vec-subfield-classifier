import os
import json
import argparse
import yaml
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import wandb


# =========================
# Utils
# =========================
def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ds_to_arrays(ds_split):
    X = np.array(ds_split["embedding"], dtype=np.float32)
    raw_labels = ds_split["label"]  # list[list[int]]
    return X, raw_labels


def derive_num_classes_and_names(
    all_raw_labels: List[List[int]],
    label_to_index_cfg: Optional[Dict[str, int]] = None,
) -> Tuple[int, List[str], Optional[Dict[str, int]]]:
    if label_to_index_cfg:
        num_classes = len(label_to_index_cfg)
        index_to_label = {v: k for k, v in label_to_index_cfg.items()}
        label_names = [index_to_label[i] for i in range(num_classes)]
        return num_classes, label_names, label_to_index_cfg
    # derive from data
    max_label = -1
    for lbls in all_raw_labels:
        if lbls:
            max_label = max(max_label, max(lbls))
    num_classes = (max_label + 1) if max_label >= 0 else 0
    label_names = [f"Label {i}" for i in range(num_classes)]
    return num_classes, label_names, None


def get_class_weights(Y: np.ndarray) -> np.ndarray:
    positives = np.sum(Y, axis=0)
    negatives = Y.shape[0] - positives
    weights = negatives / (positives + 1e-6)  # epsilon avoid div-by-zero
    return weights.astype(np.float32)


def weighted_binary_crossentropy(class_weights: np.ndarray):
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weighted_bce = bce * class_weights_tensor
        return tf.reduce_mean(weighted_bce)

    return loss


def build_mlp(
    input_size: int,
    num_classes: int,
    hidden_layers: List[int],
    activation: str,
    dropout: float,
    batch_norm: bool,
    learning_rate: float,
    class_weights: Optional[np.ndarray],
    optimizer_name: str = "adam",
):
    layers = [keras.layers.InputLayer(input_shape=(input_size,))]
    for h in hidden_layers:
        layers.append(keras.layers.Dense(h, activation=activation))
        if batch_norm:
            layers.append(keras.layers.BatchNormalization())
        if dropout and dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(num_classes, activation="sigmoid"))

    model = keras.Sequential(layers)

    if optimizer_name.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

    loss_fn = weighted_binary_crossentropy(class_weights) if class_weights is not None else "binary_crossentropy"

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(curve="ROC", multi_label=True, name="ROC_AUC"),
            keras.metrics.AUC(curve="PR", multi_label=True, name="PR_AUC"),
        ],
    )
    return model


def save_history_json(history, path: str):
    clean_history = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(path, "w") as f:
        json.dump(clean_history, f)


def plot_f1_vs_label_freq(label_counts: np.ndarray, f1_scores: List[float], label_names: List[str], out_path: str):
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(label_names)), label_counts, alpha=1.0, label="Label Count (Train)")
    plt.plot(range(len(label_names)), f1_scores, marker="o", color="red", label="F1 Score (Test)")
    plt.xticks(range(len(label_names)), label_names, rotation=90)
    plt.ylabel("Count / F1 Score")
    plt.title("Label Frequency (Train) vs F1 Score (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def maybe_wandb_init(enabled: bool, project: Optional[str], entity: Optional[str], name: Optional[str], cfg: Dict[str, Any], group: Optional[str]=None, tags: Optional[List[str]]=None):
    if not enabled:
        return None
    return wandb.init(project=project, entity=entity, name=name, config=cfg, group=group, tags=tags, reinit=True)


def build_callbacks(cfg_training: Dict[str, Any], wandb_enabled: bool):
    cbs = []
    es_cfg = cfg_training.get("early_stopping", {})
    if es_cfg.get("enabled", True):
        cbs.append(
            keras.callbacks.EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=int(es_cfg.get("patience", 5)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                restore_best_weights=True,
            )
        )
    rlrop_cfg = cfg_training.get("reduce_lr", {})
    if rlrop_cfg.get("enabled", True):
        cbs.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=rlrop_cfg.get("monitor", "val_loss"),
                factor=float(rlrop_cfg.get("factor", 0.5)),
                patience=int(rlrop_cfg.get("patience", 3)),
                verbose=1,
            )
        )
    if wandb_enabled:
        cbs.append(wandb.keras.WandbCallback(save_model=False))
    return cbs


# =========================
# Main training pipeline
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ----- Global setup -----
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    if cfg.get("mixed_precision", False):
        from tensorflow.keras import mixed_precision as mp
        policy = mp.Policy("mixed_float16")
        mp.set_global_policy(policy)

    # ----- IO paths -----
    io_cfg = cfg.get("io", {})
    save_dir = io_cfg.get("save_dir", "saved_data")
    ensure_dir(save_dir)

    def _pth(key: str, default_name: str):
        # join with save_dir if path is not absolute
        p = io_cfg.get(key, default_name)
        return p if os.path.isabs(p) else os.path.join(save_dir, p)

    model_path = _pth("model_path", "mlp_model.h5")
    history_path = _pth("history_path", "training_history.json")
    test_x_path = _pth("test_x_path", "test_embeddings.npy")
    test_y_path = _pth("test_y_path", "test_labels.npy")
    label_mapping_out = _pth("label_mapping_path", "label_mapping.json")
    f1_plot_path = _pth("f1_plot_path", "f1_vs_freq.png")

    # ----- Dataset loading -----
    ds_cfg = cfg.get("dataset", {})
    ds_name = ds_cfg.get("name", "Bocklitz-Lab/lit2vec-subfield-classifier-dataset")
    train_split_name = ds_cfg.get("train_split", "train")
    val_split_name = ds_cfg.get("val_split", "validation")
    test_split_name = ds_cfg.get("test_split", "test")
    cache_dir = ds_cfg.get("cache_dir", None)
    sample_limit = ds_cfg.get("sample_limit", None)
    shuffle_seed = ds_cfg.get("shuffle_seed", seed)

    print(f"📦 Loading dataset: {ds_name}")
    ds = load_dataset(ds_name, cache_dir=cache_dir)

    # Sanity: splits
    if train_split_name not in ds:
        raise ValueError(f"Train split '{train_split_name}' not found in dataset.")
    has_val = val_split_name in ds
    has_test = test_split_name in ds

    # Prepare splits
    X_train_split, raw_train_labels = ds_to_arrays(ds[train_split_name])
    if has_val:
        X_val_split, raw_val_labels = ds_to_arrays(ds[val_split_name])
    else:
        X_val_split, raw_val_labels = None, None
    if has_test:
        X_test_split, raw_test_labels = ds_to_arrays(ds[test_split_name])
    else:
        X_test_split, raw_test_labels = None, None

    # Optional sub-sampling (for quick experiments)
    def maybe_subsample(X, y, limit):
        if X is None or y is None or limit is None:
            return X, y
        n = min(limit, len(X))
        idx = np.arange(len(X))
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(idx)
        ss = idx[:n]
        return X[ss], [y_i for i, y_i in enumerate(y) if i in set(ss)]

    X_train_split, raw_train_labels = maybe_subsample(X_train_split, raw_train_labels, sample_limit)
    if has_val:
        X_val_split, raw_val_labels = maybe_subsample(X_val_split, raw_val_labels, sample_limit)
    if has_test:
        X_test_split, raw_test_labels = maybe_subsample(X_test_split, raw_test_labels, sample_limit)

    # ----- Labels / classes -----
    label_cfg = cfg.get("labels", {})
    label_to_index_cfg = label_cfg.get("label_to_index", None)

    all_raw_for_derivation = list(raw_train_labels)
    if raw_val_labels:
        all_raw_for_derivation += list(raw_val_labels)
    if raw_test_labels:
        all_raw_for_derivation += list(raw_test_labels)

    num_classes, label_names, label_to_index_used = derive_num_classes_and_names(
        all_raw_for_derivation, label_to_index_cfg
    )

    # MultiLabelBinarizer using consistent class set
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
    mlb.fit(raw_train_labels)
    Y_train_split = mlb.transform(raw_train_labels)
    if has_val:
        Y_val_split = mlb.transform(raw_val_labels)
    else:
        Y_val_split = None
    if has_test:
        Y_test = mlb.transform(raw_test_labels)
    else:
        Y_test = None

    print(f"✅ Using {num_classes} classes.")
    print(f"✅ Train shapes: X={X_train_split.shape}, Y={Y_train_split.shape}")
    if has_val:
        print(f"✅ Val   shapes: X={X_val_split.shape}, Y={Y_val_split.shape}")
    if has_test:
        print(f"✅ Test  shapes: X={X_test_split.shape}, Y={Y_test.shape}")

    # Save label mapping for reporting
    if label_to_index_used:
        index_to_label = {str(v): k for k, v in label_to_index_used.items()}
    else:
        index_to_label = {str(i): name for i, name in enumerate(label_names)}
        label_to_index_used = {name: i for i, name in enumerate(label_names)}
    with open(label_mapping_out, "w") as f:
        json.dump({"label_to_index": label_to_index_used, "index_to_label": index_to_label}, f)
    print(f"💾 Saved label mapping → {label_mapping_out}")

    input_size = X_train_split.shape[1]

    # ----- W&B -----
    wb_cfg = cfg.get("wandb", {})
    wb_enabled = bool(wb_cfg.get("enabled", True))
    wb_project = wb_cfg.get("project", "mlp-multilabel-example")
    wb_entity = wb_cfg.get("entity", None)
    wb_tags = wb_cfg.get("tags", ["mlp", "multilabel"])

    # ----- Cross-Validation -----
    cv_cfg = cfg.get("cross_validation", {})
    do_cv = bool(cv_cfg.get("enabled", True))
    n_folds = int(cv_cfg.get("n_folds", 5))
    cv_shuffle = bool(cv_cfg.get("shuffle", True))
    cv_seed = int(cv_cfg.get("seed", seed))

    if do_cv:
        print("\n🔁 Running K-Fold CV on train+validation...")
        if has_val:
            X_tv = np.concatenate([X_train_split, X_val_split], axis=0)
            raw_tv_labels = list(raw_train_labels) + list(raw_val_labels)
        else:
            X_tv = X_train_split
            raw_tv_labels = list(raw_train_labels)

        Y_tv = mlb.transform(raw_tv_labels)

        fold_accuracies = []
        kf = KFold(n_splits=n_folds, shuffle=cv_shuffle, random_state=cv_seed)

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tv)):
            print(f"\n===== Fold {fold_idx + 1} / {n_folds} =====")

            run = maybe_wandb_init(
                wb_enabled, wb_project, wb_entity, name=f"fold_{fold_idx+1}",
                cfg={"fold": fold_idx + 1, "n_folds": n_folds, **cfg.get("model", {}), **cfg.get("training", {})},
                group="cv",
                tags=wb_tags
            )

            X_tr, X_va = X_tv[tr_idx], X_tv[va_idx]
            Y_tr, Y_va = Y_tv[tr_idx], Y_tv[va_idx]

            class_weights_fold = get_class_weights(Y_tr) if cfg.get("class_weights", {}).get("enabled", True) else None
            print(f"✅ Computed class weights (fold {fold_idx+1}): {np.round(class_weights_fold, 2) if class_weights_fold is not None else 'disabled'}")

            model_cfg = cfg.get("model", {})
            model = build_mlp(
                input_size=input_size,
                num_classes=num_classes,
                hidden_layers=model_cfg.get("hidden_layers", [256, 256]),
                activation=model_cfg.get("activation", "relu"),
                dropout=float(model_cfg.get("dropout", 0.3)),
                batch_norm=bool(model_cfg.get("batch_norm", True)),
                learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
                class_weights=class_weights_fold,
                optimizer_name=model_cfg.get("optimizer", "adam"),
            )

            callbacks = build_callbacks(cfg.get("training", {}), wb_enabled)

            history = model.fit(
                X_tr, Y_tr,
                validation_data=(X_va, Y_va),
                batch_size=int(cfg.get("training", {}).get("batch_size", 32)),
                epochs=int(cfg.get("training", {}).get("epochs", 50)),
                verbose=1,
                callbacks=callbacks,
            )

            # Save fold history
            fold_history_path = os.path.join(save_dir, f"fold_{fold_idx+1}_history.json")
            save_history_json(history, fold_history_path)
            print(f"💾 Saved Fold {fold_idx + 1} history → {fold_history_path}")

            # Evaluate fold
            val_loss, val_binary_acc, val_roc_auc, val_pr_auc = model.evaluate(X_va, Y_va, verbose=0)
            fold_accuracies.append(val_binary_acc)

            print(f"Fold {fold_idx + 1} Metrics:")
            print(f"  🔹 Loss         : {val_loss:.4f}")
            print(f"  🔹 Binary Acc   : {val_binary_acc * 100:.2f}%")
            print(f"  🔹 ROC AUC      : {val_roc_auc:.4f}")
            print(f"  🔹 PR AUC       : {val_pr_auc:.4f}")

            if wb_enabled:
                wandb.log({
                    "val_loss": val_loss,
                    "val_binary_accuracy": val_binary_acc,
                    "val_roc_auc": val_roc_auc,
                    "val_pr_auc": val_pr_auc
                })
                wandb.finish()

        print(f"\n✅ Mean CV Binary Accuracy: {np.mean(fold_accuracies) * 100:.2f}% (+/- {np.std(fold_accuracies) * 100:.2f}%)")

    # ----- Final Train / Validate -----
    print("\n🏁 Training final model on official splits...")
    # If no explicit validation split, optionally carve from train
    tr_cfg = cfg.get("training", {})
    if not has_val and tr_cfg.get("val_split_fraction", None):
        val_frac = float(tr_cfg["val_split_fraction"])
        X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(
            X_train_split, Y_train_split, test_size=val_frac, random_state=seed, shuffle=True
        )
        has_val = True
        print(f"ℹ️ Created validation split from train: val_fraction={val_frac}")

    label_counts = np.sum(Y_train_split, axis=0)

    run = maybe_wandb_init(
        wb_enabled, wb_project, wb_entity, name=wb_cfg.get("run_name", "final_model"),
        cfg={**cfg.get("model", {}), **cfg.get("training", {})},
        group="final",
        tags=wb_tags
    )

    class_weights_final = get_class_weights(Y_train_split) if cfg.get("class_weights", {}).get("enabled", True) else None

    model_cfg = cfg.get("model", {})
    model = build_mlp(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=model_cfg.get("hidden_layers", [256, 256]),
        activation=model_cfg.get("activation", "relu"),
        dropout=float(model_cfg.get("dropout", 0.3)),
        batch_norm=bool(model_cfg.get("batch_norm", True)),
        learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
        class_weights=class_weights_final,
        optimizer_name=model_cfg.get("optimizer", "adam"),
    )

    callbacks = build_callbacks(tr_cfg, wb_enabled)

    if has_val:
        final_history = model.fit(
            X_train_split, Y_train_split,
            validation_data=(X_val_split, Y_val_split),
            batch_size=int(tr_cfg.get("batch_size", 32)),
            epochs=int(tr_cfg.get("epochs", 50)),
            verbose=1,
            callbacks=callbacks,
        )
    else:
        # fallback: validation_split if still no val
        final_history = model.fit(
            X_train_split, Y_train_split,
            validation_split=float(tr_cfg.get("val_split_fraction", 0.1)),
            batch_size=int(tr_cfg.get("batch_size", 32)),
            epochs=int(tr_cfg.get("epochs", 50)),
            verbose=1,
            callbacks=callbacks,
        )

    # Save model & history
    model.save(model_path)
    print(f"💾 Final model saved → {model_path}")
    save_history_json(final_history, history_path)
    print(f"💾 Final training history saved → {history_path}")

    # ----- Save test arrays (if available) -----
    if has_test:
        np.save(test_x_path, X_test_split)
        np.save(test_y_path, Y_test)
        print(f"💾 Test split saved → {test_x_path}, {test_y_path}")

    # ----- Final evaluation on test (if available) -----
    if has_test:
        threshold = float(tr_cfg.get("threshold", 0.5))
        Y_pred_probs = model.predict(X_test_split)
        Y_pred = (Y_pred_probs > threshold).astype(int)

        print("\n=== Final Classification Report (Test) ===")
        report = classification_report(
            Y_test, Y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
        # Print human-friendly report
        print(classification_report(Y_test, Y_pred, target_names=label_names, zero_division=0))

        f1_scores = [report[label]["f1-score"] for label in label_names]
        plot_f1_vs_label_freq(label_counts, f1_scores, label_names, f1_plot_path)
        print(f"📈 Saved F1 vs Frequency plot → {f1_plot_path}")

        if wb_enabled:
            wandb.log({"f1_vs_freq_plot": wandb.Image(f1_plot_path)})

    if wb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
