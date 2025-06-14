import streamlit as st
import joblib
import numpy as np
import lime 
import lime.lime_tabular
import pandas as pd

#Load model and data
model = joblib.load("models/best_model.pkl")
X_train,X_test,y_train,y_test = joblib.load("data/processed/train_test_split.pkl")
X = pd.DataFrame(X_train,
    columns = ['months_since_last_donation','total_volume_donated','number_of_donations','months_since_first_donation'])

#Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = X.values,
    feature_names = X.columns.tolist(),
    class_names = ['Not Donated','Donated'],
    mode = 'classification'
)

st.title("Blood Donation Predictor with Explanation")

#Input form
st.header("Enter Donor Information")

months = st.slider("Months Since Last Donation",0,60,5)
total_volume = st.slider("Total Volume Donated(ml)",0,5000,1250,step=250)
number_of_donations = st.slider("Number of Donations",0,50,4)
months_first = st.slider("Months Since First Donation",0,60,30)

#Predict
X_input = np.array([[months,total_volume,number_of_donations,months_first]])
if st.button("Predict"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction.")
    st.write("Will Donate" if prediction == 1 else "Will Not Donate")
    st.write(f"**Probability of Donation.**{proba:.2%}")

    #Explain with LIME
    exp = explainer.explain_instance(
        data_row = X_input[0],
        predict_fn = model.predict_proba,
        num_features = 4
    )

    st.subheader("Model Explanation (LIME)")
    st.components.v1.html(exp.as_html(),height=600,scrolling = True)