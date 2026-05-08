# TODO — Spam Detector (Spaaaaaam)

> See [CLAUDE.md](CLAUDE.md) for context. Tasks below are ordered roughly by priority.

## Cleanup
- [x] Pin versions in `requirements.txt` (capture from working venv: `streamlit`, `nltk`, `scikit-learn`, `numpy`, `scipy`)
- [x] Drop `pickle5` (built into Python ≥3.8)
- [x] Drop `wordcloud` and `matplotlib` from `requirements.txt` if not used at runtime
- [x] Decide on `preprocessor.pkl` — deleted; `transformText` lives in [app.py](app.py)

## Bug fixes
- [x] Use bundled `nltk_data/` instead of always calling `nltk.download(...)`
- [x] Add confidence score (shown on result + per-history-row)

## Reproducibility
- [x] Extract `main.ipynb` training pipeline into [train.py](train.py) (writes `MultinomialNB.pkl` + `Vectorizer.pkl`)
- [x] Add [tests/test_pipeline.py](tests/test_pipeline.py): preprocessing + prediction smoke tests

## Deployment
- [x] Add [Dockerfile](Dockerfile) for the Streamlit app
- [x] Add [.streamlit/config.toml](.streamlit/config.toml)
