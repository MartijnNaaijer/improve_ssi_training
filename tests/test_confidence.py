import numpy as np

from ssi.confidence import (LogisticConfidence, auroc, budget_for_accuracy, ece, fit_temperature, scale,
                            triage_curve, word_features)


def test_scale_and_temperature():
    p = np.array([0.7, 0.2, 0.1])
    assert np.isclose(scale(p, 1.0), p).all()
    assert scale(p, 2.0).max() < 0.7
    # Overconfident probabilities: the gold is the top candidate only 60% of the time.
    probs = [np.array([0.95, 0.05])] * 10
    gold = [0] * 6 + [1] * 4
    assert fit_temperature(probs, gold) > 1.5


def test_metrics():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0], dtype=bool)
    assert auroc(conf, correct) == 1.0
    assert ece(np.array([0.5, 0.5]), np.array([1, 0])) == 0.0
    curve = triage_curve(conf, correct, budgets=(0.5,))
    assert curve[0]['errors_caught'] == 1.0 and curve[0]['unchecked_acc'] == 1.0
    assert budget_for_accuracy(conf, correct, 0.99) == 0.5


def test_logistic_confidence_learns_direction():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0.3, 1.0, 500)
    y = (rng.uniform(size=500) < p1).astype(float)
    x = np.array([word_features(np.array([p, 1 - p]), 'unambig') for p in p1])
    model = LogisticConfidence().fit(x, y)
    lo, hi = model.predict(np.array([word_features(np.array([p, 1 - p]), 'unambig') for p in (0.4, 0.95)]))
    assert lo < 0.6 < 0.8 < hi
