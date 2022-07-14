# Streamlit implementation

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import fasttext



def user_input_features(symptom_df):
    symptoms = symptom_df["Clean Symptom (ID)"].tolist()
    user_symptom = st.multiselect("Pilih beberapa keluhan yang anda rasakan:", symptoms, default=['gatal'])
    text = " "
    for symptom in user_symptom:
        symptom_en = symptom_df[symptom_df["Clean Symptom (ID)"] == symptom]["Clean Symptom"].iloc[0]
        text += symptom_en + " "
        
    return text

def predict_model(text, model, k):

    list_predictions = []
    prediction = model.predict(text,k)
    for i in range(k):
        list_predictions.append(
            (prediction[0][i].replace("__label__",""), prediction[1][i])
        )
    
    return list_predictions

def show_prediction(prediction, disease_df):
    
    prediction_1 = prediction[0][0]
    prediction_2 = prediction[1][0]
    prediction_3 = prediction[2][0]

    prediction_1 = disease_df[disease_df["Disease Slug"] == prediction_1]["Disease (ID)"].iloc[0]
    prediction_2 = disease_df[disease_df["Disease Slug"] == prediction_2]["Disease (ID)"].iloc[0]
    prediction_3 = disease_df[disease_df["Disease Slug"] == prediction_3]["Disease (ID)"].iloc[0]

    score_1 = "{:.1%}".format(prediction[0][1])
    score_2 = "{:.1%}".format(prediction[1][1])
    score_3 = "{:.1%}".format(prediction[2][1])

    description_1 = disease_df[disease_df["Disease (ID)"] == prediction_1]["Description (ID)"].iloc[0]
    description_2 = disease_df[disease_df["Disease (ID)"] == prediction_2]["Description (ID)"].iloc[0]
    description_3 = disease_df[disease_df["Disease (ID)"] == prediction_3]["Description (ID)"].iloc[0]

    precuation_1 = str(str(disease_df[disease_df["Disease (ID)"] == prediction_1]["Precaution_1 (ID)"].iloc[0]) + ", " +
                    str(disease_df[disease_df["Disease (ID)"] == prediction_1]["Precaution_2 (ID)"].iloc[0]) + ", " + 
                    str(disease_df[disease_df["Disease (ID)"] == prediction_1]["Precaution_3 (ID)"].iloc[0]) + ", " + 
                    str(disease_df[disease_df["Disease (ID)"] == prediction_1]["Precaution_4 (ID)"].iloc[0])) 

    precuation_2 = str(str(disease_df[disease_df["Disease (ID)"] == prediction_2]["Precaution_1 (ID)"].iloc[0]) + ", " +
                    str(disease_df[disease_df["Disease (ID)"] == prediction_2]["Precaution_2 (ID)"].iloc[0]) + ", " + 
                    str(disease_df[disease_df["Disease (ID)"] == prediction_2]["Precaution_3 (ID)"].iloc[0]) + ", " + 
                    str(disease_df[disease_df["Disease (ID)"] == prediction_2]["Precaution_4 (ID)"].iloc[0]))

    precuation_3 = str(str(disease_df[disease_df["Disease (ID)"] == prediction_3]["Precaution_1 (ID)"].iloc[0]) + ", " +
                    str(disease_df[disease_df["Disease (ID)"] == prediction_3]["Precaution_2 (ID)"].iloc[0]) + ", " + 
                    str(disease_df[disease_df["Disease (ID)"] == prediction_3]["Precaution_3 (ID)"].iloc[0]) + ", " + 
                    str(disease_df[disease_df["Disease (ID)"] == prediction_3]["Precaution_4 (ID)"].iloc[0]))
    
    st.subheader(f"{prediction_1}, {score_1}")
    st.write(description_1)
    st.write(f"Rekomendasi penyembuhan: {precuation_1}")
    st.subheader(f"{prediction_2}, {score_2}")
    st.write(description_2)
    st.write(f"Rekomendasi penyembuhan: {precuation_2}")
    st.subheader(f"{prediction_3}, {score_3}")
    st.write(description_3)
    st.write(f"Rekomendasi penyembuhan: {precuation_3}")



def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # prepare variables
    symptom_df = pd.read_csv('data/symptom_id.csv')
    disease_df = pd.read_csv('data/disease_id.csv')
    model = fasttext.load_model('model/model.ftz')

    st.title("Get-Health Application")
    st.write("Aplikasi yang mampu memprediksi kemungkinan penyakit yang dialami dari gejala pasien")
    st.write('---')

    st.header('Masukan gejala yang dirasakan')
    text = user_input_features(symptom_df)
    predict = st.button("Predict")
    st.write('---')

    st.header("Kemungkinan penyakit yang anda miliki")
    if predict:
        print(text)
        prediction = predict_model(text, model, 3)
        show_prediction(prediction, disease_df)
    st.write('---')
    st.write('Get-Health Application - Heatlhcare 4 DTS Pro Academy 2022 Batch 2 - Machine Learning with Tensorflow')

if __name__ == "__main__":
    main()

