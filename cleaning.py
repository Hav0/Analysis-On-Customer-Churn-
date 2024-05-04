import pandas as pd

# read the data 
data = pd.read_csv('telecom_churn.csv')

# check for na values in the dataset
data_null = pd.isnull(data)
data.dropna()

# remove unneccesary variables from the dataset
data = data.drop(['AccountWeeks', 'DataPlan', 'ContractRenewal', 'CustServCalls', 'OverageFee', 'RoamMins'], axis = 1)
print((data))
print(data.describe())






