import pandas as pd 
data=pd.read_csv("Churn_Modelling.csv")

data_subset=data.iloc[:50]
data_subset=data_subset.drop(["RowNumber","CustomerId","Exited","Surname"],axis=1)
print(data_subset.head())
data_subset.to_csv("data_subset.csv", index=False)

