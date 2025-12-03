import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Load the saved model
# ---------------------------
@st.cache_resource
def load_model():
    with open("iris_dt.pkl", "rb") as f:    # iris_dt.pkl must be in same folder as app.py
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Iris Flower Classifier (Decision Tree)")

st.write("Enter flower measurements to predict the Iris species.")

# These names must match the order of columns used during training
sepal_length = st.number_input("Sepal length (cm)",  min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal width (cm)",   min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal length (cm)",  min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal width (cm)",   min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict Species"):
    # Make a 2D array as model expects: [[f1, f2, f3, f4]]
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted species: **{prediction}**")