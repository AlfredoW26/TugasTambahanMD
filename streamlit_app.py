import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

def main():
  st.title('Machine Learning App')
  st.info('This app will predict your obesity level!')

  # input data by user
  gender = st.selectbox('Gender',('Male','Female'))
  age = st.slider('Age', min_value = 0, max_value = 75, value = 21)
  height = st.slider('Height', min_value = 0.00, max_value = 2.00, value = 1.62)
  weight = st.slider('Weight', min_value = 0.00, max_value = 140.00, value = 64)
  family_history_with_overweight = st.selectbox('family_history_with_overweight',('yes','no'))
  FAVC = st.selectbox('FAVC',('yes','no')) 
  FCVC = st.slider('FCVC', min_value = 0.00, max_value = 3.00, value = 2.00)
  NCP = st.slider('NCP', min_value = 0.00, max_value = 3.00, value = 2.00)
  CAEC = st.selectbox('CAEC',('no','Sometimes','Frequently','Always')) 
  SMOKE = st.selectbox('SMOKE',('yes','no'))
  CH2O = st.slider('CH2O', min_value = 0.00, max_value = 3.00, value = 2.00)
  SCC = st.selectbox('SCC',('yes','no'))
  FAF = st.slider('FAF', min_value = 0.00, max_value = 3.00, value = 2.00)
  TUE = st.slider('TUE', min_value = 0.00, max_value = 3.00, value = 2.00)
  CALC = st.selectbox('CALC',('no','Sometimes','Frequently','Always')) 
  CAEC = st.selectbox('CAEC',('no','Sometimes','Frequently','Always')) 



  
  # Input Data for Program
  user_input = [erythema, scaling, definite_borders, itching, koebner_phenomenon, polygonal_papules, follicular_papules, oral_mucosal_involvement, knee_and_elbow_involvement, scalp_involvement, family_history, melanin_incontinence, eosinophils_infiltrate, PNL_infiltrate,
                fibrosis_papillary_dermis, exocytosis, acanthosis, hyperkeratosis, parakeratosis, clubbing_rete_ridges, elongation_rete_ridges, thinning_suprapapillary_epidermis, spongiform_pustule, munro_microabcess, focal_hypergranulosis, disappearance_granular_layer, 
                vacuolisation_damage_basal_layer, spongiosis, saw_tooth_appearance_retes, follicular_horn_plug, perifollicular_parakeratosis, inflammatory_mononuclear_infiltrate, band_like_infiltrate, age]

  model_filename = 'trained_model.pkl'
  model = load_model(model_filename)
  prediction = predict_with_model(model, user_input)
  st.write('The prediction output is: ', prediction)

if __name__ == "__main__":
  main()
