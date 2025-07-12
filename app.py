import streamlit as st
import joblib

model = joblib.load("model/emotion_model.pkl")

st.title("Emotion Detector 😄😭😡")
user_input = st.text_input("Enter a sentence:")
if st.button("Detect Emotion"):
    result = model.predict([user_input])
    st.success(f"Detected Emotion: {result[0].capitalize()}")
