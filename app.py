import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import json
import plotly.express as px

# Load the animation from the JSON file  
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Hero Section
st.markdown(
    """
    <style>
        .hero {
            background-color: #f7f7f7;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .hero h1 {
            color: #4CAF50;
            font-size: 2.5rem;
        }
        .hero p {
            color: #555;
            font-size: 1.2rem;
        }
    </style>
    <div class="hero">
        <h1>Iris Flower Prediction App</h1>
        <p>Experience a modern and interactive way to classify Iris flowers!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom Fonts
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Poppins', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to navigate.")
feature = st.sidebar.radio("Select Feature", ["Prediction", "Visualization"])

if feature == "Prediction":
    # Input fields for the user
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Input Flower Features:")
    sepal_length = st.number_input("Sepal Length (cm):", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
    sepal_width = st.number_input("Sepal Width (cm):", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
    petal_length = st.number_input("Petal Length (cm):", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
    petal_width = st.number_input("Petal Width (cm):", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")

    # Prediction button
    if st.button("Predict"):
        # Prepare input for the model
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Display the result
        st.write(f"**Predicted Class:** {prediction}")

elif feature == "Visualization":
    st.header("Visualization Page")

    # Example Plotly Chart
    data = {
        "sepal_length": [5.1, 4.9, 6.3, 5.8],
        "sepal_width": [3.5, 3.0, 3.3, 2.7],
        "class": ["Iris-setosa", "Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    }
    fig = px.scatter(
        data, x="sepal_length", y="sepal_width", color="class",
        title="Sepal Dimensions by Class"
    )
    st.plotly_chart(fig)

    # Advanced Layout with Columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Input Features")
        sepal_length = st.slider("Sepal Length", 0.0, 10.0, step=0.1)
        sepal_width = st.slider("Sepal Width", 0.0, 10.0, step=0.1)

    with col2:
        st.header("Results")
        st.metric("Predicted Class", "Iris-setosa")

# Add Lottie Animation
lottie_iris = load_lottiefile("Animation - 1735629737763.json")
st_lottie(lottie_iris, height=300, key="iris")

# Footer
st.markdown(
    """
    <footer style='text-align: center; padding: 10px; margin-top: 50px;'>
        <hr style="border: none; border-top: 1px solid #eee;">
        <p style="color: #888; font-size: 0.9rem;">
            © 2024 Iris Flower Prediction App | Created with ❤️ using Streamlit
        </p>
    </footer>
    """,
    unsafe_allow_html=True
)

# Style Buttons
st.markdown(
    """
    <style>
    div[data-testid="stButton"] button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a styled button
if st.button("Styled Predict Button"):
    st.write("Styled button clicked!")

# Include Bootstrap or Tailwind CSS
st.markdown(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    <div class="container text-center">
        <h1 class="display-4">Iris Flower Prediction</h1>
        <p class="lead">A modern UI for flower classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)
