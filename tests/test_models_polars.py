from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from era_pl import models


@pytest.mark.parametrize("rus", [False, True], ids=["adaboost", "rusboost"])
@pytest.mark.parametrize("frame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
def test_perfect_first_learner_has_unit_weight(rus, frame):
    data = frame({"x": [0, 0, 1, 1], "y": [0, 0, 1, 1]})

    fit = models.rusboost(data, ["x"], "y", size=3, rus=rus, maxdepth=1)

    assert len(fit.weak_learners) == 1
    np.testing.assert_array_equal(fit.alpha, [1.0])
    np.testing.assert_array_equal(fit.predict(data[["x"]]), [0, 0, 1, 1])
    np.testing.assert_array_equal(
        fit.predict_proba(data[["x"]]), [[1, 0], [1, 0], [0, 1], [0, 1]]
    )
    np.testing.assert_array_equal(fit.predict_proba(data[["x"]].head(1)), [[1, 0]])


@pytest.fixture
def training_data():
    return pl.DataFrame({"x": range(8), "y": [0] * 4 + [1] * 4})


def mock_trees(monkeypatch, predictions):
    learners = [Mock(predict=Mock(return_value=np.asarray(p))) for p in predictions]
    factory = Mock(side_effect=learners)
    monkeypatch.setattr("sklearn.tree.DecisionTreeClassifier", factory)
    return factory, learners


@pytest.mark.parametrize("rus", [False, True], ids=["adaboost", "rusboost"])
def test_perfect_later_learner_replaces_previous_learners(monkeypatch, training_data, rus):
    response = training_data["y"].to_numpy()
    factory, learners = mock_trees(monkeypatch, [[0] * 6 + [1] * 2, response])
    # The first learner is perfect on the RUS sample, but misses two full-data rows.
    monkeypatch.setattr(models, "rus_sample", lambda *a, **k: np.array([0, 1, 6, 7]))

    fit = models.rusboost(
        training_data, ["x"], "y", size=3, rus=rus, learn_rate=0.5
    )

    assert factory.call_count == 2
    assert fit.weak_learners == [learners[1]]
    np.testing.assert_array_equal(fit.alpha, [1.0])
    np.testing.assert_array_equal(fit.predict(training_data[["x"]]), response)
    np.testing.assert_array_equal(fit.predict_proba(training_data[["x"]])[:, 1], response)


@pytest.mark.parametrize("rus", [False, True], ids=["adaboost", "rusboost"])
def test_unusable_learners_are_skipped_or_stop_training(monkeypatch, training_data, rus):
    imperfect = [0] * 6 + [1] * 2
    bad = [1] * 4 + [0] * 4
    predictions = [bad, imperfect, bad] if rus else [imperfect, bad]
    factory, learners = mock_trees(monkeypatch, predictions)

    fit = models.rusboost(training_data, ["x"], "y", size=3, rus=rus)

    assert factory.call_count == len(predictions)
    assert fit.weak_learners == [learners[1 if rus else 0]]
    np.testing.assert_allclose(fit.alpha, [np.log(3)])
    np.testing.assert_array_equal(fit.predict(training_data[["x"]]), imperfect)
    np.testing.assert_array_equal(
        fit.predict_proba(training_data[["x"]])[:, 1], imperfect
    )


@pytest.mark.parametrize("rus", [False, True], ids=["adaboost", "rusboost"])
def test_no_usable_learner_raises(rus):
    data = pl.DataFrame({"x": [0] * 8, "y": [0] * 4 + [1] * 4})

    with pytest.raises(ValueError, match="No usable weak learner was fitted"):
        models.rusboost(data, ["x"], "y", size=3, rus=rus, maxdepth=1)


@pytest.mark.parametrize("rus", [False, True], ids=["adaboost", "rusboost"])
def test_zero_weighted_error_is_not_a_perfect_learner(monkeypatch, training_data, rus):
    factory, learners = mock_trees(monkeypatch, [[0] * 6 + [1] * 2] * 3)
    original_update = models.w_update
    updates = 0

    def update(prediction, response, w, learn_rate):
        nonlocal updates
        updates += 1
        if updates == 1:
            # Simulate underflow: the two misclassified rows now have zero weight.
            return np.array([1 / 6] * 4 + [0, 0] + [1 / 6] * 2), np.log(3), 0.25
        return original_update(prediction, response, w, learn_rate)

    monkeypatch.setattr(models, "w_update", update)

    fit = models.rusboost(training_data, ["x"], "y", size=3, rus=rus)

    assert factory.call_count == (3 if rus else 2)
    assert fit.weak_learners == [learners[0]]
    np.testing.assert_allclose(fit.alpha, [np.log(3)])


def test_skipping_rus_learner_preserves_weight_normalization(monkeypatch, training_data):
    factory, learners = mock_trees(
        monkeypatch,
        [[0] * 6 + [1] * 2, [1] * 4 + [0] * 4, [0] * 4 + [1] * 4],
    )
    monkeypatch.setattr(models, "rus_sample", lambda *a, **k: np.arange(8))
    weights = np.array([1 / 12] * 4 + [1 / 4] * 2 + [1 / 12] * 2)
    weights[0] += np.finfo(float).eps
    original_update = models.w_update
    updates = 0

    def update(prediction, response, w, learn_rate):
        nonlocal updates
        updates += 1
        if updates == 1:
            return weights.copy(), np.log(3), 0.25
        return original_update(prediction, response, w, learn_rate)

    monkeypatch.setattr(models, "w_update", update)

    fit = models.rusboost(training_data, ["x"], "y", size=3)

    assert factory.call_count == 3
    assert fit.weak_learners == [learners[2]]
    np.testing.assert_array_equal(
        learners[2].fit.call_args.kwargs["sample_weight"],
        weights / weights.sum(),
    )


@pytest.mark.parametrize("rus", [False, True], ids=["adaboost", "rusboost"])
@pytest.mark.parametrize("response", [0, 1])
def test_single_class_keeps_constant_probability_model(rus, response):
    data = pl.DataFrame({"x": [0, 1, 2], "y": [response] * 3})

    fit = models.rusboost(data, ["x"], "y", size=3, rus=rus)

    assert isinstance(fit, models.ConstantProbModel)
    np.testing.assert_array_equal(fit.predict(data[["x"]]), [response] * 3)
    np.testing.assert_array_equal(
        fit.predict_proba(data[["x"]]), [[1 - response, response]] * 3
    )


@pytest.mark.parametrize(
    ("rus", "expected_alpha", "fractional_probs"),
    [
        (False, [1.4722194895832204, 0.9574529110789962, 0.8490513926041217,
                 0.6046297798579887], [0.5977490430570757] * 4),
        (True, [1.4722194895832204, 0.9574529110789962, 0.5568653295371012,
                0.6097996651912734], [0.5789276494032534] + [0.7337699982470612] * 3),
    ],
    ids=["adaboost", "rusboost"],
)
def test_ordinary_predictions_match_before_fix(rus, expected_alpha, fractional_probs):
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] + 0.25 * X[:, 1] + rng.normal(size=20) > 0).astype(np.int8)
    data = pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})
    # These weights and predictions were captured from the unmodified checkout.
    expected = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1])
    probabilities = expected.astype(float)
    probabilities[[11, 13, 14, 19]] = fractional_probs

    fit = models.rusboost(
        data, ["x1", "x2"], "y", size=4, rus=rus, maxdepth=1,
        learn_rate=0.5, random_state=2021,
    )

    assert len(fit.weak_learners) == 4
    np.testing.assert_allclose(fit.alpha, expected_alpha, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(fit.predict(X), expected)
    np.testing.assert_allclose(
        fit.predict_proba(X), np.column_stack([1 - probabilities, probabilities]),
        rtol=1e-12, atol=1e-12,
    )
