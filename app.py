from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st
import pickle
model=load_model('model.h5')
scaler=pickle.load(open('scaler.pkl','rb'))
geo_ohe=pickle.load(open('ohe_Geo.pkl','rb'))
gender_le=pickle.load(open('le_Gender.pkl','rb'))
st.title("Customer Retention Prediction")
st.write("Please enter the details of the customer")

age=st.slider("Age",18,100)
CreditScore=st.number_input("Credit Score")
Geography=st.selectbox("Country",['France','Spain','Germany'])  
Tenure=st.number_input("Tenure")
Balance=st.number_input("Balance")
NumOfProducts=st.number_input("Number of Products")
HasCrCard=st.selectbox("Has Credit Card",[0,1])
isActiveMember=st.selectbox("Is Active Member",[0,1])
EstimatedSalary=st.number_input("Estimated Salary")
Gender=st.selectbox("Gender",["Male","Female"])
# st.write(Gender)

Gender=gender_le.transform([Gender])

geo=geo_ohe.transform([[Geography]])
geo_df=pd.DataFrame(geo,columns=geo_ohe.get_feature_names_out())

list=[CreditScore,age,Gender,Tenure,Balance,NumOfProducts,HasCrCard,isActiveMember,EstimatedSalary]
columns=['CreditScore', 'Gender' ,'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember', 'EstimatedSalary']
input_data=pd.DataFrame([list],columns=columns)
input_data=pd.concat([input_data,geo_df],axis=1)
input_data_scaled=scaler.transform(input_data)
prediction=model.predict(input_data_scaled)
if st.button("Predict"):
    if prediction>0.5:
        st.write("Customer will leave the bank")
    else:
        st.write("Customer will not leave the bank")



