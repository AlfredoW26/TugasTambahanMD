import streamlit as st
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

st.title('🎈 App Name')

st.write('Hello world!')
