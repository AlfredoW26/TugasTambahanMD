import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('trained_model.pkl')
loaded_encoder = joblib.load('encoding.pkl')
loaded_scaler = joblib.load('scaling.pkl')

def input_to_df(input):
    df = pd.DataFrame([input], columns=[
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ])
    return df

def encode(df):
    for column in df.columns:
        if column in loaded_encoder:
            df[column] = loaded_encoder[column].transform(df[column])
    return df
    
def normalize(df):
    df = df.astype(float)
    df_scaled = loaded_scaler.transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)

def main():
    st.title('Machine Learning App')
    st.info('This app will predict your obesity level!')

    with st.expander('**Data**'):
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        st.write(df)

    with st.expander('**Data Visualization**'):
        st.scatter_chart(data=df, x='Height', y='Weight', color='NObeyesdad')

    gender = st.selectbox('Gender', ('Male', 'Female'))
    age = st.slider('Age', min_value=0, max_value=75, value=21)
    height = st.slider('Height', min_value=0.50, max_value=2.00, value=1.62)  
    weight = st.slider('Weight', min_value=10.0, max_value=140.0, value=64.0) 
    family_history_with_overweight = st.selectbox('Family History With Overweight', ('yes', 'no'))
    FAVC = st.selectbox('Frequent Consumption of High Caloric Food', ('yes', 'no')) 
    FCVC = st.slider('Vegetable Consumption Frequency', min_value=0.0, max_value=3.0, value=2.0)
    NCP = st.slider('Number of Meals per Day', min_value=0.0, max_value=3.0, value=2.0)
    CAEC = st.selectbox('Consumption of Food Between Meals', ('no', 'Sometimes', 'Frequently', 'Always')) 
    SMOKE = st.selectbox('Smoking Habit', ('yes', 'no'))
    CH2O = st.slider('Daily Water Intake', min_value=0.0, max_value=3.0, value=2.0)
    SCC = st.selectbox('Caloric Drinks Consumption', ('yes', 'no'))
    FAF = st.slider('Physical Activity Frequency', min_value=0.0, max_value=3.0, value=2.0)
    TUE = st.slider('Time Using Technology', min_value=0.0, max_value=3.0, value=2.0)
    CALC = st.selectbox('Alcohol Consumption', ('no', 'Sometimes', 'Frequently', 'Always')) 
    MTRANS = st.selectbox('Transportation Mode', ('Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'))

    user_input = [gender, age, height, weight, family_history_with_overweight, 
                  FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]
    
    df_input = input_to_df(user_input)

    st.write('Data input by user')
    st.write(df_input)

    # Preprocessing
    df_input = encode(df_input)
    df_input = normalize(df_input)
    
    try:
        feature_names = model.feature_names_in_
    except AttributeError:
        feature_names = df_input.columns
    
    df_input = df_input[feature_names]
    
    prediction = model.predict(df_input)[0]
    prediction_proba = model.predict_proba(df_input)

    df_prediction_proba = pd.DataFrame(prediction_proba, columns=[
        'Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
        'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III'
    ])

    st.write('Obesity Prediction Probability')
    st.write(df_prediction_proba)
    st.write('The predicted output is:', prediction)

if __name__ == "__main__":
    main()
