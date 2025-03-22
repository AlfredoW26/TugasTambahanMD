import streamlit as st
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

st.title('Machine Learning App')

model = load_model("trained_model.pkl")
