# DrivenInsights: Used Car Price Prediction

A machine learning project that estimates the fair market value of a used vehicle from its make, year, mileage, and other key specs — served through a Flask web app and JSON API.

## Overview

Pricing a used car fairly is hard to do by eye — small differences in mileage, horsepower, or body type can swing the value by thousands. This project trains and compares several regression models on used-car listing data, picks the best performer, and serves live price predictions through a simple web form or API.

The original coursework version used PySpark over a cleaned used-car dataset. This version keeps that notebook as a reference while adding a production-style scikit-learn training pipeline and a lightweight Flask app for actually using the model.

## How It Works

1. **Training** — `train_model.py` loads the dataset and trains three candidate models: **Ridge Regression**, **Random Forest**, and **Gradient Boosting**.
2. **Model selection** — Each model is evaluated with **R², MAE, and RMSE**, and the best-performing one is chosen automatically.
3. **Persistence** — The winning model (plus its feature list) is serialized with `joblib` into the `models/` folder, so it only needs to be trained once.
4. **Serving** — `app.py` loads the saved model and exposes it two ways:
   - A simple web form where you pick make, year, mileage, horsepower, fuel type, and body type, and get back an estimated price
   - A JSON API (`POST /predict`) for programmatic use

## Features Used

- Make
- Model year
- Mileage
- Horsepower
- Fuel type
- Body type

## Tech Stack

- **scikit-learn** — model training and evaluation (Ridge, Random Forest, Gradient Boosting)
- **joblib** — model serialization
- **Flask** — web app and API
- **pandas** — data handling
- **Jupyter Notebook** — original exploratory analysis (PySpark-based, kept for reference)

## Project Structure

```
├── data/
│   └── used_cars_sample.csv     # sample dataset for public/reviewer use
├── docs/
│   └── Source.txt               # link to the original full dataset
├── notebooks/
│   └── used_cars.ipynb          # original PySpark exploratory notebook
├── models/                      # generated locally when training runs (git-ignored)
├── train_model.py               # scikit-learn training pipeline
├── app.py                       # Flask app + prediction API
└── requirements.txt
```

## Getting Started

### Train the model

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

### Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### Use the API directly

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","year":2021,"mileage":45000,"horsepower":210,"fuel_type":"Hybrid","body_type":"SUV"}'
```

Returns:
```json
{"estimated_price": 24500.0}
```

## Notes

- A compact sample dataset (`data/used_cars_sample.csv`) is included so anyone can run this end-to-end without downloading the full source data. The original dataset link is documented in `docs/Source.txt`.
- Flask's `debug=True` mode is enabled for local development convenience — turn this off before deploying anywhere public.

## Contributors

Built collaboratively as a team project.
