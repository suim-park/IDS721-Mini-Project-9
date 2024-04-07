import streamlit as st
from transformers import pipeline

# Load a pre-trained model pipeline for sentiment analysis or text classification
model = pipeline(
    "text-classification", model="textdetox/xlmr-large-toxicity-classifier"
)

st.title("Toxicity Detection")
user_input = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    # Model prediction
    result = model(user_input)
    # Simplify to just display 'Toxic' or 'Not Toxic'
    if result[0]["label"] == "LABEL_1":  # Adjust based on your model's output
        st.write("Toxic")
    else:
        st.write("Not Toxic")
