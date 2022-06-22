import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
plt.ylabel("Average Total Traffic acroos 4 Bridges in NYC")
plt.xticks(rotation = 'vertical')
plt.show()
