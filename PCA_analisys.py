from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import kstest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Data = pd.read_csv(r"C:\Users\jayson3xxx\Desktop\Training_DataSeTs\eda_data.csv", low_memory=False)
Data.dropna()
Data.drop_duplicates()
fact_cols = Data.select_dtypes(include='object').apply(lambda x: pd.factorize(x)[0]).columns
Data[fact_cols] = Data[fact_cols].apply(lambda x: pd.factorize(x)[0]).copy()
plt.show()
y_np = Data['avg_salary'].to_numpy()
x_df = Data.drop(labels= ['avg_salary'] , axis = 1 ).copy()
x_np = x_df.to_numpy().copy()
plt.show()
scaler = StandardScaler().fit(x_np)
x_st = scaler.transform(x_np)
pca = PCA()
x_pca = pca.fit_transform(x_st)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()
pca_new = PCA(n_components=25)
x_new = pca_new.fit_transform(x_st)
x_train , x_test , y_train , y_test = train_test_split(x_new , y_np )
Model = GradientBoostingRegressor().fit(x_train , y_train)
print("Количество правильных ответов на обучающей выборке  : ", Model.score(x_train,y_train))
print("Количество правильных ответов на тестовой выборке  : ", Model.score(x_test,y_test))