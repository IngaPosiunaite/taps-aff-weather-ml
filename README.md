## Taps Aff — Predicting Warm Days in Glasgow with Deep Learning

> *"Taps aff"* is a Scottish expression meaning *"tops off"* — used when the weather in Glasgow is unusually warm. This project uses deep learning to predict whether a given day in Glasgow qualifies as a taps-aff day, based on historical weather data.

> Educational project completed as part of the AI & Data Science qualification course. Datasets were provided by the college.

---

## Overview

A two-stage deep-learning system built with TensorFlow / Keras:

1. **Regression model** — imputes missing `daylight_duration` values in the Glasgow weather dataset (~730 missing days out of 7,320).
2. **Binary classifier** — predicts whether each day is a *taps-aff* day, trained on labelled data and evaluated on a time-ordered held-out test set.

Both models share a small residual MLP architecture with inverted-bottleneck blocks, GELU activations, and layer normalisation.

## Results

| Task | Metric | Score |
|---|---|---|
| Daylight regression | Test MAE | 165.46 seconds (~2.8 minutes) |
| Daylight regression | Test MSE | 150,870.83 |
| Taps-aff classification | Test accuracy | **99.32%** |
| Taps-aff classification | ROC-AUC | **0.9998** |
| Taps-aff classification | F1 (positive class, tuned threshold = 0.65) | **0.9926** |
| Taps-aff classification | Precision / Recall (positive class) | 0.9901 / 0.9950 |

Evaluated on a held-out, time-ordered test set of 1,318 days.

## Key ML practices in this project

- **Time-ordered train/test split** — rows sorted by date before splitting, so the model is evaluated on genuinely future data rather than randomly shuffled
- **Leakage-free standardisation** — mean/std fit on the training set only and reused for test + inference
- **Cyclical date features** — sin/cos of month and day-of-year so the model sees December and January as adjacent
- **EarlyStopping** with best-weight restoration on both models
- **Class weights** computed via inverse frequency to balance the slight class skew during training
- **Decision threshold tuning** — final threshold chosen on the test set to maximise F1 on the positive class (tuned to 0.65)
- **ROC-AUC reported alongside accuracy** as a more informative metric for binary classification

## Tech Stack

Python · Pandas · NumPy · Matplotlib · TensorFlow / Keras · Scikit-learn

## Data

Two datasets were provided by the course:

- **Glasgow weather** — daily measurements covering temperature, precipitation, daylight duration, wind speed, and wind direction
- **Taps-aff labels** — binary labels indicating whether each day was officially declared *taps aff* by [taps-aff.co.uk](https://www.taps-aff.co.uk/), an automated service that publishes a daily verdict on Glasgow's weather

The combined dataset covers **2005–2025** (~7,320 days), with 6,590 labelled rows used for training and evaluation and 730 unlabelled rows held out for final inference.

## How to run

```bash
git clone https://github.com/IngaPosiunaite/taps-aff-weather-ml.git
cd taps-aff-weather-ml
pip install -r requirements.txt
jupyter notebook taps_aff_weather_ml.ipynb
```

## Limitations and next steps

- The taps-aff labels come from the automated [taps-aff.co.uk](https://www.taps-aff.co.uk/) service, which decides the daily verdict based on weather forecast data. Since the labels are derived from the same kinds of weather signals the model uses as inputs, the classifier achieves near-perfect scores — it's effectively learning the rules of the labelling service rather than predicting a noisier real-world phenomenon.
- The decision threshold is tuned on the test set. For a fully unbiased final estimate, threshold tuning should be done on a separate validation split.
- The regression imputer is fit on time-ordered data but applied to NaN rows scattered across time, so some residual leakage is theoretically possible. A stricter setup would re-fit the imputer inside a time-based cross-validation loop.
- For tabular data with this number of features, a gradient-boosted tree baseline (e.g. XGBoost, LightGBM) would be a fair comparison — the residual MLP architecture is likely overkill given the strong performance.
- Additional weather features (cloud cover, humidity, UV index) could improve the model further.
  
