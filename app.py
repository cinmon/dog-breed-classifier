import streamlit as st
from inference import load_model, predict_image
import os

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")
st.title("Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload an image of a dog!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Save the uploaded file to a temporary location
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        # f.write(uploaded_file.getbuffer())
        f.write(uploaded_file.read())

    # Load the model
    model = load_model()
    st.write("Thinking...")

    # Predict the breed of the dog in the uploaded image
    predicted_class, confidence = predict_image(temp_file_path, model)

    # Display the results
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    parsed_predicted_class = predicted_class.split("-")[1].replace("_", " ").title()

    st.success(f"I think this is a **{parsed_predicted_class}**!")

    # Clean up the temporary file
    os.remove(temp_file_path)