# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:31:56 2024

@author: VARUN
"""

import numpy as np
import pickle
import streamlit as st

loaded_mdel = pickle.load(open('diabetes_model.sav','rb'))

def diabetes_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_mdel.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return('The person is not diabetic')
    else:
      return('The person is diabetic')
  

def main():
    
    st.title("Diabetes Prediction Web App")
    
    
    Pregnancies = st.text_input("Enter No of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure Value")
    SkinThickness = st.text_input("SkinThickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")
    
    # Code for prediction
    diagnosis=''
    
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__ == "__main__":
    main()
    
    
