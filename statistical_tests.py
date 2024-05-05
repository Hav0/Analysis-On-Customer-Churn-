import sys 
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from cleaning import * 
from data_visualizations import *

# Chi-squared test for Churn Status and the continous variable = MonthlyCharge
contingency = pd.crosstab(data_without_outliers['Churn'], data_without_outliers['MonthlyCharge'])
print(contingency)
chi_squared_result = stats.chi2_contingency(contingency).pvalue
print("Chi-Squared pvalue for Churn Status and Monthly Charge Amounts: ", chi_squared_result)

# two-sample Test for Churn Status (1 or 0) and MonthlyCharge
monthlycharge_churn = data_with_churn['MonthlyCharge']
monthlycharge_notchurn = data_without_churn['MonthlyCharge']
two_samptest_pvalue = stats.ttest_ind(monthlycharge_churn, monthlycharge_notchurn, equal_var = True, alternative = 'two-sided').pvalue
print("Two-Sample T-test pvalue for Churn Status and Monthly Charge Amounts: ", two_samptest_pvalue)

# mann whitney U-test for Churn Status (1 or 0) and Data Usage 
datausage_churn = data_with_churn['DataUsage']
datausage_notchurn = data_without_churn['DataUsage']
mannwhitney_pvalue = stats.mannwhitneyu(datausage_churn, datausage_notchurn).pvalue
print("Mann-Whitney U-Test pvalue for Churn Status and Data Usage Amounts: ", mannwhitney_pvalue)
