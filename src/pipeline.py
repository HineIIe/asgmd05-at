import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.ingest import load_data
from src.evaluate import evaluate_model
from src.preprocess import build_preprocessor
from src.train import tune_logistic_regression
from src.preprocess import FeatureEngineer

RANDOM_STATE = 42


def run_pipeline():

    print("========== STARTING PIPELINE ==========")

    # =========================
    # 1 Load Data
    # =========================
    df = load_data("data/train.csv")

    # =========================
    # 2 Feature Engineering 
    # =========================
    
    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["Cabin_num"] = df["Cabin"].str.split("/").str[1]
    df["Side"] = df["Cabin"].str.split("/").str[2]

    df["Cabin_num"] = pd.to_numeric(df["Cabin_num"], errors="coerce")

    # =========================
    # 3 Define Features
    # =========================
    categorical_features = [
        "HomePlanet", "CryoSleep", "Destination",
        "VIP", "Deck", "Side"
    ]

    numerical_features = [
        "Age", "RoomService", "FoodCourt",
        "ShoppingMall", "Spa", "VRDeck", "Cabin_num"
    ]

    X = df.drop(columns=["Transported"])
    y = df["Transported"].astype(int)

    # =========================
    # 4 Split
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # =========================
    # 5 Preprocessor
    # =========================
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # =========================
    # 6 Hyperparameter Tuning (Optuna)
    # =========================
    best_params = tune_logistic_regression(X_train, y_train, preprocessor)

    # =========================
    # 7 Pipeline
    # =========================
    pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),   
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(random_state=RANDOM_STATE))
    ])

    pipeline.set_params(**best_params)

    # =========================
    # 8 Train
    # =========================
    pipeline.fit(X_train, y_train)

    # =========================
    # 9 Evaluate
    # =========================
    evaluate_model(pipeline, X_val, y_val)

    # =========================
    # 10 Save
    # =========================
    os.makedirs("models", exist_ok=True)

    with open("models/pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("\nPipeline saved successfully!")
    print("========== PIPELINE FINISHED ==========")


if __name__ == "__main__":
    run_pipeline()