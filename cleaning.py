import pandas as pd

# read the data 
data = pd.read_csv('telecom_churn.csv')

# check for na values in the dataset
data_null = pd.isnull(data)
data.dropna()

# remove unneccesary variables from the dataset
data = data.drop(['AccountWeeks', 'DataPlan', 'ContractRenewal', 'CustServCalls', 'OverageFee', 'RoamMins'], axis = 1)

# exclude monthly charge amounts that are greater than $98 (potential outliers)
threshold = 98 
data_without_outliers = data[data['MonthlyCharge'] <= threshold]
#print((data_without_outliers))
#print(data_without_outliers.describe())






