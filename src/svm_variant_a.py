"""Practical work 9, level A: visualize SVM kernels and the effect of C."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


def load_data():
    """Generate reproducible 2D data and split before any scaling."""
    X, y = make_moons(n_samples=320, noise=0.22, random_state=RANDOM_STATE)
    return train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )


def build_models():
    """Scaler is fitted on training features only, preventing leakage."""
    return {
        "linear_C1": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
        "rbf_C1": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "rbf_C005": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=0.05, gamma="scale")),
        "rbf_C100": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=100.0, gamma="scale")),
    }


def fit_and_score(models, X_train, X_test, y_train, y_test):
    result = {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        result[key] = {
            "kernel": model.named_steps["svc"].kernel,
            "C": model.named_steps["svc"].C,
            "gamma": str(model.named_steps["svc"].gamma),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy": round(accuracy_score(y_test, model.predict(X_test)), 4),
            "test_f1": round(f1_score(y_test, model.predict(X_test)), 4),
            "n_support_vectors": int(model.named_steps["svc"].support_.size),
        }
    return result


def plot_boundary(ax, model, X_train, y_train, title, score):
    """Plot in original 2D coordinates; the pipeline scales internally."""
    step = 0.025
    x_low, x_high = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_low, y_high = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_low, x_high, step), np.arange(y_low, y_high, step)
    )
    mesh = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(mesh).reshape(xx.shape)
    decision = model.decision_function(mesh).reshape(xx.shape)
    ax.contourf(xx, yy, predictions, levels=[-0.5, 0.5, 1.5],
                colors=["#dbeafe", "#ffedd5"], alpha=0.85)
    ax.contour(xx, yy, decision, levels=[0], colors=["#111827"], linewidths=1.8)
    for cls, color, marker in [(0, "#2563eb", "o"), (1, "#ea580c", "^")]:
        mask = y_train == cls
        ax.scatter(X_train[mask, 0], X_train[mask, 1], c=color,
                   marker=marker, s=28, edgecolors="white",
                   linewidth=0.35, alpha=0.9)
    supports = model.named_steps["svc"].support_
    ax.scatter(X_train[supports, 0], X_train[supports, 1], s=78,
               facecolors="none", edgecolors="#111827",
               linewidths=0.9, label="Опорные векторы")
    ax.set(xlabel="Признак 1 (исходная шкала)", ylabel="Признак 2 (исходная шкала)")
    ax.set_title(f"{title}\nTest accuracy = {score:.3f}; опорных = {len(supports)}", fontsize=11)
    ax.grid(alpha=0.12)
    ax.legend(loc="lower left", fontsize=8)


def save_comparison(models, metrics, X_train, y_train, names, title, output_path):
    fig, axes = plt.subplots(1, len(names),
                             figsize=(6 * len(names), 4.4),
                             constrained_layout=True)
    for ax, (key, label) in zip(np.atleast_1d(axes), names):
        plot_boundary(ax, models[key], X_train, y_train,
                      label, metrics[key]["test_accuracy"])
    fig.suptitle(title, fontweight="bold", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=155, bbox_inches="tight")
    plt.close(fig)


def run_experiment(output_dir: Path | str = "results"):
    output_dir = Path(output_dir)
    X_train, X_test, y_train, y_test = load_data()
    models = build_models()
    metrics = fit_and_score(models, X_train, X_test, y_train, y_test)
    save_comparison(
        models, metrics, X_train, y_train,
        [("linear_C1", "Линейное ядро, C=1"), ("rbf_C1", "RBF-ядро, C=1")],
        "Сравнение ядер SVM", output_dir / "kernels_comparison.png"
    )
    save_comparison(
        models, metrics, X_train, y_train,
        [("rbf_C005", "RBF, C=0.05"), ("rbf_C1", "RBF, C=1"),
         ("rbf_C100", "RBF, C=100")],
        "Влияние параметра C на RBF SVM", output_dir / "c_effect.png"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"dataset": "make_moons", "n_samples": 320, "noise": 0.22,
                    "random_state": RANDOM_STATE, "train_size": len(X_train),
                    "test_size": len(X_test), "models": metrics},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return models, metrics, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    _, scores, _ = run_experiment(root / "results")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
