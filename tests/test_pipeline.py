import os
import pickle
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import transformText  # noqa: E402


@pytest.fixture(scope="module")
def model():
    with open(os.path.join(ROOT, "MultinomialNB.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def vectorizer():
    with open(os.path.join(ROOT, "Vectorizer.pkl"), "rb") as f:
        return pickle.load(f)


def test_transform_text_produces_stems():
    cleaned = transformText("FREE WIN!!! click")
    tokens = cleaned.split()
    assert tokens, "transformText should not return empty string for non-empty input"
    assert all(t.isalnum() for t in tokens)
    assert "free" in tokens or "win" in tokens or "click" in tokens
    assert "!" not in cleaned


def test_transform_text_empty_input():
    assert transformText("") == ""
    assert transformText(None) == ""


def test_pipeline_predicts_label(model, vectorizer):
    cleaned = transformText("FREE WIN!!! click here to claim your prize now")
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)
    assert pred.shape == (1,)
    assert int(pred[0]) in (0, 1)

    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba.max()) <= 1.0
