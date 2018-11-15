
import tensorflow as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io

# Loading data and converting it into a .csv file
data_xls = pd.read_excel('Sarcos.xls', 'Sheet1', index_col=None)
data_xls.to_csv('Sarcos_csv.csv', encoding='utf-8')

data1 = data_xls.iloc[:,1] # reading from index 1 in the csv file
print(data1)

scaler = StandardScaler()
scaled_data1 = scaler.fit_transform(data1.values.reshape(-1, 1))

##### Plotting the data #####
plt.figure(figsize=(12,7),frameon=False, facecolor='brown',edgecolor='blue')
plt.title('Data column 1')
plt.xlabel('Time steps t')
plt.ylabel('Position of joint 1')
plt.plot(scaled_data1,label='pos,joint1')
plt.legend()
plt.show()
