import streamlit as st
import pandas as pd
import pickle


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Spaceship Titanic Prediction",
    layout="wide"
)


# =========================
# Load Pipeline (Cached)
# =========================
@st.cache_resource
def load_pipeline():
    with open("models/pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


pipeline = load_pipeline()


# =========================
# Title
# =========================
st.title("🚀 Spaceship Titanic Transport Prediction")
st.write("ASG 05 MD - **Anang Tantowi**")


# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Passenger Information")

HomePlanet = st.sidebar.selectbox(
    "Home Planet",
    ["Earth", "Europa", "Mars"]
)

CryoSleep = st.sidebar.selectbox(
    "CryoSleep",
    [True, False]
)

Destination = st.sidebar.selectbox(
    "Destination",
    ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]
)

Age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

VIP = st.sidebar.selectbox(
    "VIP",
    [True, False]
)


st.sidebar.header("Cabin Information")

Cabin = st.sidebar.text_input(
    "Cabin (example: B/123/P)",
    "B/0/P"
)

PassengerGroup = st.sidebar.number_input(
    "Passenger Group",
    min_value=1,
    value=1
)


st.sidebar.header("Spending")

RoomService = st.sidebar.number_input("Room Service", min_value=0.0, value=0.0)
FoodCourt = st.sidebar.number_input("Food Court", min_value=0.0, value=0.0)
ShoppingMall = st.sidebar.number_input("Shopping Mall", min_value=0.0, value=0.0)
Spa = st.sidebar.number_input("Spa", min_value=0.0, value=0.0)
VRDeck = st.sidebar.number_input("VR Deck", min_value=0.0, value=0.0)


# =========================
# Prediction Button
# =========================
if st.sidebar.button("Predict"):

    # Create input dataframe (RAW DATA — pipeline handles everything)
    input_data = pd.DataFrame({
        "PassengerId": [f"{PassengerGroup}_01"],
        "HomePlanet": [HomePlanet],
        "CryoSleep": [CryoSleep],
        "Cabin": [Cabin],
        "Destination": [Destination],
        "Age": [Age],
        "VIP": [VIP],
        "RoomService": [RoomService],
        "FoodCourt": [FoodCourt],
        "ShoppingMall": [ShoppingMall],
        "Spa": [Spa],
        "VRDeck": [VRDeck],
        "Name": ["Streamlit Passenger"]
    })

    st.subheader("Input Data")
    st.dataframe(input_data)


    # =========================
    # Prediction (PIPELINE ONLY)
    # =========================
    with st.spinner("Running model..."):

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]


    # =========================
    # Output
    # =========================
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Passenger was Transported")
    else:
        st.error("❌ Passenger was NOT Transported")

    st.metric(
        label="Transport Probability",
        value=f"{probability:.2%}"
    )


# =========================
# Footer
# =========================
st.markdown("---")
st.write("Model: Logistic Regression (Pipeline)")
st.write("Course: Model Deployment")