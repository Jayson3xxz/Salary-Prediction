from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import kstest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
Data = pd.read_csv(r"C:\Users\jayson3xxx\Desktop\Training_DataSeTs\eda_data.csv", low_memory=False)
Data.dropna()
Data.drop_duplicates()
print(Data.info())
drop_cols = ['Job Description' , 'Salary Estimate' , 'Company Name' , 'Location' , 'Headquarters' , 'Type of ownership' , 'Sector' , 'Revenue' , 'Competitors' , 'min_salary' , 'max_salary' , 'company_txt' , 'job_state',
             'same_state' , 'job_simp' , 'desc_len']
Data = Data.drop(Data[drop_cols] , axis = 1).copy()
print(Data.info())
fact_cols = ['Size' , 'Job Title' , 'Industry' , 'seniority']
Data[fact_cols] = Data[fact_cols].apply(lambda x: pd.factorize(x)[0]).copy()
fig = plt.figure(figsize=(8,8))
# sns.heatmap(Data.corr(method = 'spearman') , xticklabels = Data.corr(method = 'spearman').columns ,
#             yticklabels = Data.corr(method = 'spearman').columns,cmap = 'coolwarm' , center = 0 , annot = True)
plt.show()
drop_cols2 = ['Unnamed: 0','Size' , 'Founded' , 'Industry' , 'employer_provided' , 'R_yn' , 'excel','num_comp']
Data.drop(labels= drop_cols2 , axis =1 , inplace=True)
print(Data.info())

fig = plt.figure()
y_np = Data['avg_salary'].to_numpy().copy()
Y_NP = np.sort(y_np)
print(kstest(Y_NP , method = 'approx' , N = 100 , cdf = 'norm'))
IQR = Data['avg_salary'].quantile(0.75) - Data['avg_salary'].quantile(0.25)
Data = Data[(Data['avg_salary'] > 1.5*IQR - Data['avg_salary'].quantile(0.25)) & (Data['avg_salary'] < 1.5*IQR + Data['avg_salary'].quantile(0.75))].copy()
y_np = Data['avg_salary'].to_numpy()
x_df = Data.drop(labels= ['avg_salary'] , axis = 1 ).copy()
sns.heatmap(x_df.corr(method = 'spearman') , xticklabels = x_df.corr(method = 'spearman').columns ,
           yticklabels = x_df.corr(method = 'spearman').columns,cmap = 'coolwarm' , center = 0 , annot = True)
x_np = x_df.to_numpy().copy()
plt.show()
scaler = MinMaxScaler().fit(x_np)
x_st = scaler.transform(x_np)
x_train , x_test , y_train ,y_test = train_test_split(x_st , y_np)
fig = plt.figure(figsize=(9,9))
