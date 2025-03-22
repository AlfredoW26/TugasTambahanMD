import streamlit as st
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

st.title('Machine Learning App')

st.write('Hello world!')
