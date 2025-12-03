import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

@st.cache_resource
def train_model():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model, iris.feature_names, iris.target_names

model, feature_names, target_names = train_model()

st.title("Iris Flower Classifier (Decision Tree)")

st.write("Enter flower measurements to predict the Iris species.")

sepal_length = st.number_input("Sepal length (cm)",  0.0, 10.0, 5.1)
sepal_width  = st.number_input("Sepal width (cm)",   0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length (cm)",  0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal width (cm)",   0.0, 10.0, 0.2)

if st.button("Predict"):
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_idx = model.predict(x)[0]
    st.success(f"Predicted species: {target_names[pred_idx]}")
