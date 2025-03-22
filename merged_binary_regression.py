import streamlit as st
import pandas as pd
import pickle
from keras import backend as K
from tensorflow.keras.models import load_model

model=load_model('model.h5')




scaler=pickle.load(open('scaler.pkl','rb'))
geo_ohe=pickle.load(open('ohe_Geo.pkl','rb'))
gender_le=pickle.load(open('le_Gender.pkl','rb'))
def binary_classification():
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
def salary_regression():
    def mse(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))
    salary_model=load_model("salary_model.h5", custom_objects={'mse': mse})
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
    
    geo_encoded=ohe_geo.transform([[geography]])
    geo_encoded=pd.DataFrame(geo_encoded,columns=ohe_geo.categories_[0])
    input_data=pd.concat([input_data,geo_encoded],axis=1)

    input_data = pd.DataFrame(scale.transform(input_data), columns=input_data.columns)
    prediction=salary_model.predict(input_data)

    st.write("Predicted value",prediction)

st.sidebar.title("Navigation")
page=st.sidebar.radio("Choose a model:",["Binary Classification","Salary Prediction"])

if page=="Binary Classification":
    binary_classification()
elif page=="Salary Prediction":
    salary_regression()
