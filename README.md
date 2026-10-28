# ✈️ Flight Delay Predictor | Spark, GraphFrames and MLlib

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C) ![License](https://img.shields.io/badge/License-MIT-purple)

## 📌 Overview

A cluster-computing project (INT315) that analyses U.S. airline on-time data and predicts whether a flight will arrive more than 15 minutes late.

1. **Spark DataFrames** clean about one million flight records and compute delay statistics.
2. **GraphFrames** (the Python API for the GraphX model) turns airports into nodes and routes into links, then computes PageRank, degrees, connected components and shortest paths.
3. **Spark MLlib** trains Logistic Regression and a 50-tree Random Forest using flight attributes plus the graph features.
4. A **Streamlit app** runs the trained models so anyone can enter a flight and get a prediction. The trained trees and weights are exported to `model_export.json`, so the app needs no Spark.

## 🚀 Key Features

- Predicts only when you press **Predict delay risk**
- Side-by-side Random Forest and Logistic Regression scores
- Live input checks (route not in the data, month with no data, early hours)
- Same-trip charts: every departure hour and every airline
- Airport network map with PageRank, search and a route overlay
- Delay explorer by airline, month and hour, plus plain-language model explanations

## 🛠️ Technologies Used

- **Cluster computing:** Apache Spark 3.5 (PySpark), GraphFrames, Spark MLlib
- **App:** Streamlit, Altair, pandas
- **Data:** U.S. DOT on-time records for 2015 ([Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)), random 20% sample

## ⚙️ Run the app locally

```bash
git clone https://github.com/<your-username>/Flight-Delay-Prediction-System.git
cd Flight-Delay-Prediction-System
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. `model_export.json` is included, so there is nothing to train first.

## 🔁 Retrain the models (optional)

1. Download the dataset from Kaggle and put `flights.csv`, `airports.csv` and `airlines.csv` in a `datasets/` folder.
2. `pip install pyspark==3.5.1 graphframes==0.6 matplotlib pandas numpy` (Java is required for Spark).
3. Run `Spark_Flight_Delay_Notebook.ipynb` top to bottom. The last cell rewrites `model_export.json`.
4. Make sure the jar in the notebook matches your Scala version (`_2.12` or `_2.13`).

## 🌐 Get a live link (free)

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub and choose **New app**.
3. Pick the repo, branch `main` and main file `app.py`, then deploy. You get a public `*.streamlit.app` link.

## 📂 Project Structure

```text
Flight-Delay-Prediction-System/
├── app.py                           # Streamlit app (predictor, network, explorer, models)
├── model_export.json                # Trained trees, weights, airports, routes, stats
├── Spark_Flight_Delay_Notebook.ipynb# Spark + GraphFrames + MLlib pipeline and export cell
├── requirements.txt                 # App dependencies
├── .streamlit/config.toml           # Theme
├── .gitignore
├── LICENSE
└── README.md
```

## 📊 Model Performance (20% test split)

| Model | AUC | Accuracy | F1 | Precision | Recall | Train time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6277 | 0.5915 | 0.6377 | 0.7549 | 0.5915 | 15.5 |
| Random Forest | 0.6429 | 0.5856 | 0.6325 | 0.7596 | 0.5856 | 30.3 |

Precision, recall and F1 are weighted over both classes. Class weights were used, so the app's score is not a literal chance of delay: 50% or more is where the model predicts "late".

**Limits:** only pre-departure information is used (no weather or earlier-flight delays), the data covers 2015, and the run was in Spark local mode on one machine.

## 📜 License

MIT License.
