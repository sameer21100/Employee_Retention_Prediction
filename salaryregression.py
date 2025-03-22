from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st
import pickle

from keras import backend as K

def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


model = load_model("salary_model.h5", custom_objects={'mse': mse})

st.title("Predict salary: ")

ohe_geo=pickle.load(open("ohe_Geo.pkl","rb"))
le_gender=pickle.load(open("le_Gender.pkl","rb"))
scale=pickle.load(open("salary_scaler.pkl","rb"))

creditScore=st.number_input("Enter the credit Score")
geography=st.selectbox("Choose the Geography",ohe_geo.categories_[0])
gender=st.selectbox("Enter the gender",le_gender.classes_)
age=st.slider("Enter you age",18,99)
tenure=st.number_input("Enter you tenure")
balance=st.number_input("Enter you balance")
numOfProducts=st.number_input("Enter no. of Products")
hasACrCard=st.selectbox("has a credit card",[0,1])
isActiveMember=st.selectbox("is active member",[0,1])
exited=st.selectbox(" Has exited " ,[0,1])


input_data=pd.DataFrame({
    'CreditScore':[creditScore],
    'gender':[le_gender.transform([gender])[0]],
    'age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'numOfProducts':[numOfProducts],
    'hasCreditCard':[hasACrCard],
    'IsActiveMember':[isActiveMember],
    'Exited':[exited]
})
# 
geo_encoded=ohe_geo.transform([[geography]])
geo_encoded=pd.DataFrame(geo_encoded,columns=ohe_geo.categories_[0])
input_data=pd.concat([input_data,geo_encoded],axis=1)

input_data = pd.DataFrame(scale.transform(input_data), columns=input_data.columns)
prediction=model.predict(input_data)

st.write(prediction)

st.write(input_data)
