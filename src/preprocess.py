import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()


        df["Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else "Unknown")
        df["Cabin_num"] = df["Cabin"].apply(lambda x: x.split("/")[1] if pd.notna(x) else -1)
        df["Side"] = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else "Unknown")

        df["Cabin_num"] = pd.to_numeric(df["Cabin_num"], errors="coerce")

        df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
        df["Group_size"] = df.groupby("Group")["Group"].transform("count")
        df["Solo"] = (df["Group_size"] == 1).astype(int)

        df["LastName"] = df["Name"].apply(lambda x: x.split()[-1] if pd.notna(x) else "Unknown")
        df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")

        spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        df["TotalSpending"] = df[spending_cols].sum(axis=1)
        df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
        df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

        for col in spending_cols:
            df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

        df["Age_group"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 30, 50, 100],
            labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"]
        )

        df["Age_missing"] = df["Age"].isna().astype(int)
        df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

        return df
    
    
    
def build_preprocessor(categorical_features, numerical_features):

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipeline, categorical_features),
        ("num", numerical_pipeline, numerical_features)
    ])

    return preprocessor