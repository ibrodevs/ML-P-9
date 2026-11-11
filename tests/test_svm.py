from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.svm_variant_a import build_models, fit_and_score, load_data, run_experiment


def test_dataset_split_and_scaling():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape == (240, 2)
    assert X_test.shape == (80, 2)
    assert len(y_train) + len(y_test) == 320
    for model in build_models().values():
        assert isinstance(model, Pipeline)
        assert isinstance(model.named_steps["standardscaler"], StandardScaler)


def test_kernel_comparison_and_c_effect():
    X_train, X_test, y_train, y_test = load_data()
    scores = fit_and_score(build_models(), X_train, X_test, y_train, y_test)
    assert scores["rbf_C1"]["test_accuracy"] > scores["linear_C1"]["test_accuracy"]
    assert scores["rbf_C005"]["n_support_vectors"] != scores["rbf_C100"]["n_support_vectors"]


def test_exports_all_required_results(tmp_path: Path):
    _, results, _ = run_experiment(tmp_path)
    assert len(results) == 4
    for name in ["kernels_comparison.png", "c_effect.png", "metrics.json"]:
        assert (tmp_path / name).stat().st_size > 100
