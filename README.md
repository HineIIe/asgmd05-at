# Spaceship Titanic

## Overview
This project builds a **machine learning pipeline** using **Logistic Regression** to predict whether a passenger was transported or not

The focus of this project is not only model performance, but also:
- Clean project structure
- Reproducible pipeline
- Proper use of scikit-learn tools

---


## Project Workflow

### Pipeline Architecture


Load Data
↓
Feature Engineering
↓
Split
↓
Build Preprocessor
↓
Optuna finds best params
↓
Build Pipeline
↓
Apply best params
↓
Train
↓
Evaluate
↓
Save


---

## Project Structure


project_root/
│
├── data/
│ ├── raw/ # Original dataset
│  
│
├── models/
│ └── pipeline.pkl
│
│
├── src/
│ ├── __init__.py
│ ├── ingest.py 
│ ├── preprocess.py 
│ ├── pipeline.py
│ └── evaluate.py
│ └── train.py
│
├── requirements.txt
└── README.md


---

## Features Used

### Categorical Features
- HomePlanet
- CryoSleep
- Destination
- VIP
- Deck
- Side

### Numerical Features
- Age
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck
- Cabin_num

---

## Model

### Logistic Regression
- Simple and interpretable baseline model
- Works well with structured/tabular data

---

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- ROC-AUC
- Classification Report
- Confusion Matrix

---

## How to Run

### 1. Install dependencies
###    pip install -r requirements.txt
### 2. Run pipeline
###    python -m src.pipeline



## Output

### After running, the model will be saved to:
### models/pipeline.pkl



# Author
# Anang Tan