import matplotlib.pyplot as plt 
import seaborn as sns 
from cleaning import * 

# Boxplot of Churn status and Average Monthly Bill 
sns.set() 
plt.figure(figsize=(7,5)) 
light_palette = sns.color_palette("husl", n_colors=2, desat=0.7)

threshold = 98
sns.boxplot(x = 'Churn', y = 'MonthlyCharge', data = data_without_outliers, palette= light_palette)
plt.ylabel('Monthly Charge Amounts ($)')
plt.title('Boxplot of Montlhy Charge Amounts by Churn Status')
plt.savefig('Figures/Monthly_Amounts_Churn_Status_boxplot.png')

#Boxplot of Churn Satus and Average Daytime Minutes per Month
plt.figure(figsize=(7,5)) 
sns.boxplot(x = 'Churn', y = 'DayMins', data = data_without_outliers, palette= light_palette)
plt.ylabel('Average Daytime Minutes (Per Month)')
plt.title('Boxplot of Average Daytime Minutes by Churn Status')
plt.savefig('Figures/Daytime_Minutes_Churn_Status_boxplot.png')

# histogram of Churned Customers and Monthly Bill Averages
plt.figure(figsize=(7,5)) 
data_with_churn = data_without_outliers[data_without_outliers['Churn'] == 1]
sns.histplot(data = data_with_churn, x = 'MonthlyCharge')
plt.xlabel('Monthly Charge Amounts ($)')
plt.title('Histogram of Monthly Charge Amounts for Churned Customers')
plt.savefig('Figures/Monthly_Charge_Churn_histogram.png')

# hitogram of Non-Churned Customers and Monthly Bill Averages 
plt.figure(figsize=(7,5)) 
data_without_churn = data_without_outliers[data_without_outliers['Churn'] == 0]
sns.histplot(data = data_without_churn, x = 'MonthlyCharge')
plt.xlabel('Monthly Charge Amounts ($)')
plt.title('Histogram of Monthly Charge Amounts for Non-Churned Customers')
plt.savefig('Figures/Monthly_Charge_NonChurn_histogram.png')

