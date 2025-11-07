# Fake News Detection

A compact, end-to-end pipeline to classify news as real or fake using text preprocessing, TF-IDF features, and three models: Logistic Regression, SVM, and LSTM.

## Quick Start

- Clone and install:
  ```bash
  git clone https://github.com/monaii/Fake-News-Detection.git
  cd Fake-News-Detection
  pip install -r requirements.txt
  ```
- Run with traditional ML models:
  ```bash
  python simple_main.py
  ```
- Run full pipeline (includes LSTM):
  ```bash
  python main.py
  ```

## What You Get
- Cleaned text features via TF-IDF
- Three trained models (LR, SVM, LSTM)
- Evaluation metrics and saved plots in `results/`

## Repo Layout
```
src/            # data loading, preprocessing, evaluation, models
data/           # datasets
results/        # generated visualizations
main.py         # full pipeline
simple_main.py  # traditional ML pipeline
```

## Performance
- Target F1 ≥ 0.92 achieved by all models on the provided dataset.

## More Details
See `FINAL_PROJECT_REPORT.md` for the full technical report.