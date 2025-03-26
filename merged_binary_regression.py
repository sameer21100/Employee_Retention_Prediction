import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.inspection import permutation_importance

model=load_model('model.h5',compile=True)

scaler=pickle.load(open('scaler.pkl','rb'))
geo_ohe=pickle.load(open('ohe_Geo.pkl','rb'))
gender_le=pickle.load(open('le_Gender.pkl','rb'))


def compute_feature_importance(model, X, y):
    result = permutation_importance(model, X, y, scoring="accuracy", n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    return importance_df

def plot_feature_importance(importance_df):
    st.subheader("Feature Importance (Permutation Importance)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="coolwarm", ax=ax)
    ax.set_title("Feature Importance (Permutation Importance)")
    st.pyplot(fig)

def churn_distribution(df):
    st.subheader("Churn Rate Analysis")
    fig, ax = plt.subplots()
   
    sns.countplot(x=df["Churn Prediction"], ax=ax, palette=["green", "red"])

    ax.set_xticklabels(["Not Churned", "Churned"])
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Count")
    ax.set_title("Churn Distribution")

    st.pyplot(fig)

def binary_classification():
    st.title("Customer Retention Prediction")
    if "batch_mode" not in st.session_state:
        st.session_state.batch_mode=False
    if "single_mode" not in st.session_state:
        st.session_state.single_mode=False
    
    
    if st.button(("Upload a Batch File")):
        st.session_state.batch_mode=True
    if st.button("Enter a single Person Data"):
        st.session_state.single_mode=True
    if st.session_state.batch_mode:
        batch_file=st.file_uploader("Upload csv file",type=["csv"])
        if batch_file is not None:
            st.success("Batch file uploaded Successfully")
            if batch_file:
                df = pd.read_csv(batch_file)
                st.write("Data Preview:", df.head()) 

            if batch_file:
                df["Gender"] = gender_le.transform(df["Gender"])  # label Encode Gender
                geo_encoded = geo_ohe.transform(df[["Geography"]])  # One-hot encoded Geography
                geo_df = pd.DataFrame(geo_encoded, columns=geo_ohe.get_feature_names_out())

                df = df.drop(columns=["Geography"])
                df = pd.concat([df, geo_df], axis=1)
                input_data_scaled = scaler.transform(df)
                predictions = model.predict(input_data_scaled)
                df["Churn Prediction"] = (predictions > 0.5).astype(int)
                st.write("Predictions:", df.head())
                churn_distribution(df)
            
        
    elif st.session_state.single_mode:
        st.subheader("Enter data manually")
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
        important_feature=compute_feature_importance(model, df.drop(columns=["Churn Prediction"]), df["Churn Prediction"])
        plot_feature_importance(important_feature)
       

       

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
