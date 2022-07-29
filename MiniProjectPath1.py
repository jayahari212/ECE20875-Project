import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestRegressor


''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pd.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pd.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_1.to_string()) #This line will print out your data
dataset_1['Total']                = pd.to_numeric(dataset_1['Total'].replace(',','', regex=True))


# Individual Data Frames for each month

df_april = dataset_1.iloc[0:30].reset_index(drop = True)
df_may = dataset_1.iloc[30:61].reset_index(drop = True)
df_june = dataset_1.iloc[61:91].reset_index(drop = True)
df_july = dataset_1.iloc[91:122].reset_index(drop = True)
df_august = dataset_1.iloc[122:153].reset_index(drop = True)
df_september = dataset_1.iloc[153:183].reset_index(drop = True)
df_october = dataset_1.iloc[183:214].reset_index(drop = True)

# Data Frames for Total Traffic for each month
total_april_traffic = df_april['Total'].to_frame().rename(columns={"Total": "April Total"})
total_may_traffic = df_may['Total'].to_frame().rename(columns={"Total": "May Total"})
total_june_traffic = df_june['Total'].to_frame().rename(columns={"Total": "June Total"})
total_july_traffic = df_july['Total'].to_frame().rename(columns={"Total": "July Total"})
total_august_traffic = df_august['Total'].to_frame().rename(columns={"Total": "August Total"})
total_september_traffic = df_september['Total'].to_frame().rename(columns={"Total": "September Total"})
total_october_traffic = df_october['Total'].to_frame().rename(columns={"Total": "October Total"})

# Concatenating Data frames to determine average traffic
df = pd.concat([total_april_traffic, total_may_traffic, total_june_traffic, total_july_traffic, total_august_traffic, total_september_traffic, total_october_traffic], axis = 1)
df_average = df.mean(axis=1)


# Plotting Average Total Traffic between months of April and October by Day of Month
x_axis = np.linspace(1, 31, 31)

plt.plot(x_axis, df_average, 'x')
plt.title("Average Total Traffic between April and October v Day of Month in 2016")
plt.xlabel("Day of Month")
plt.ylabel("Average Total Traffic across 4 Bridges in NYC")
plt.xticks(rotation = 'vertical')
plt.show()



dataset_1.info()
#creating histogram with all bikers across each bridge each day
#between April and October
bBridge = dataset_1['Brooklyn Bridge'].tolist()
mBridge = dataset_1['Manhattan Bridge'].tolist()
wBridge = dataset_1['Williamsburg Bridge'].tolist()
qBridge = dataset_1['Queensboro Bridge'].tolist()
dates = dataset_1['Date'].tolist()
dataset_2 = pd.DataFrame({'Brooklyn': bBridge, 'Manhattan': mBridge, 'Williamsburg': wBridge,
                            'Queensboro': qBridge}, index = dates)
ax = dataset_2.plot.bar(rot = 0)


#making headers etc. for three-plot figure
fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (20, 5))
sns.regplot(x = dataset_1['High Temp'], y = dataset_1['Total'], ax = ax1)
ax1.set(title = "Relationship between daily high temperature and total bikers")
sns.regplot(x = dataset_1['Low Temp'], y = dataset_1['Total'], ax = ax2)
ax2.set(title = "Relationship between daily low temperature and total bikers")
sns.regplot(x = dataset_1['Precipitation'], y = dataset_1['Total'], ax = ax3)
ax3.set(title = "Relationship between daily precipitation and total bikers")

corr = dataset_1.corr()
plt.figure(figsize = (10, 8))

#data pre-processing and splitting dataset into training and testing
X = dataset_1[['High Temp', 'Low Temp', 'Precipitation']]
Y = dataset_1[['Total']]
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

#regular linear regr
print("\nLinear Regr Stats:")
regr = LinearRegression(fit_intercept = True)
regr.fit(X_Train, Y_Train)
Y_Pred = regr.predict(X_Test)
print("Intercept: \n", regr.intercept_)
print('Coefficients: \n', regr.coef_)
print('MSE: %.2f' % mean_squared_error(Y_Test, Y_Pred))
print('R^2: %.2f' % r2_score(Y_Test, Y_Pred))

plt.scatter(X_Test.iloc[:,0], Y_Test, label = 'True')
plt.scatter(X_Test.iloc[:,0], Y_Pred, label = 'Predicted')
plt.legend()

#normalizing the values for normalized linear regr
scaler = StandardScaler()
X_Train_Norm = scaler.fit_transform(X_Train)
Y_Train_Norm = scaler.fit_transform(Y_Train)

X_Test_Norm = scaler.fit_transform(X_Test)
Y_Test_Norm = scaler.fit_transform(Y_Test)

linRegrNorm = LinearRegression(fit_intercept = True)
linRegrNorm.fit(X_Train_Norm, Y_Train_Norm)

Y_linPred_Norm = linRegrNorm.predict(X_Test_Norm)
Y_linPred = scaler.inverse_transform(Y_linPred_Norm)
Y_linPred_Norm = linRegrNorm.predict(X_Test_Norm)
Y_linPred = scaler.inverse_transform(Y_linPred_Norm)
regrNorm = LinearRegression(fit_intercept = True)
regrNorm.fit(X_Test_Norm, Y_Test_Norm)
print("\nLinear Regr for Normalized Vals:")
print("Intercept: \n", regrNorm.intercept_)
print('Coefficients: \n', regrNorm.coef_)
print("Normalized MSE: %.2f" % mean_squared_error(Y_Test, Y_linPred))
print("Normalized R^2: %.2f" % r2_score(Y_Test, Y_linPred))

"""
#random forest regr and caluclations
print("\nRandom Forest Regr:")
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_Train_Norm, Y_Train_Norm.ravel())
Y_Random_Forest_Pred_Norm = random_forest_model.predict(X_Test_Norm)
Y_Random_Forest_Pred = scaler.inverse_transform(Y_Random_Forest_Pred_Norm)
print('MSE: %.2f' % mean_squared_error(Y_Test, Y_Random_Forest_Pred))
print('R^2: %.2f' % r2_score(Y_Test, Y_Random_Forest_Pred))
"""

#ridge regr and calculations
print("\nRidge Regr:")
ridgeReg = linear_model.Ridge(alpha = 10, fit_intercept = True)
ridgeReg.fit(X_Train_Norm, Y_Train_Norm)
Y_ridgePred_Norm = ridgeReg.predict(X_Test_Norm)
Y_ridgePred = scaler.inverse_transform(Y_ridgePred_Norm)
print('Intercept: \n', ridgeReg.intercept_)
print('Coefficients: \n', ridgeReg.coef_)
print('MSE: %.2f' % mean_squared_error(Y_Test, Y_ridgePred))
print('R^2: %.2f' % r2_score(Y_Test, Y_ridgePred))
print('R^2: %.2f' % r2_score(Y_Test, Y_ridgePred))

#heat map:
sns.heatmap(corr, annot = True, annot_kws = {'size': 10})
