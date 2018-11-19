
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io

################## Windowing function ####################################
def windowed_data(data,window_size):
    X = []
    y = []
    i = 0

    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

        i += 1
    assert len(y) == len(X)
    return X,y
##########################################################################
# Loading data and converting it into a .csv file
data_xls = pd.read_excel('Sarcos.xls', 'Sheet1', index_col=None)
#data_xls.to_csv('Sarcos_csv.csv', encoding='utf-8')

data1 = data_xls.iloc[:,1] # reading from index 1 in the csv file
#print(data1)

scaler = StandardScaler()
scaled_data1 = scaler.fit_transform(data1.values.reshape(-1, 1))

################## Plotting the data ######################################
"""plt.figure(figsize=(12,7),frameon=False, facecolor='brown',edgecolor='blue')
plt.title('Data column 1')
plt.xlabel('Time steps t')
plt.ylabel('Position of joint 1')
plt.plot(scaled_data1,label='pos,joint1')
plt.legend()
plt.show()
"""
############################################################################
# Defining hyperparameters, THEY SHOULD BE MODIFIED
batch_size = 10         # number of windows passed at once
window_size = 10        # number of data points we want to predict
hidden_layers = 250     # number of hidden layers in the LSTM
clip_margin = 4         # used to clip gradients above/below its margins
learning_rate = 0.0001  # way to optimize the loss function
epochs = 200            # number of iterations the model needs to make (forward and back propagation)
############################################################################
# Placeholders
inputs = tf.placeholder(tf.float32, [batch_size,window_size, 1])
outputs = tf.placeholder(tf.float32, [batch_size, 1])
############################################################################
# Splitting into training and validation data

X, y = windowed_data(data1,window_size)
size_of_80p = int(len(scaled_data1)*0.8)

X_train = np.array(X[:size_of_80p])              #we want 80 % to be training data
y_train = np.array(y[:size_of_80p])

X_test = np.array(X[size_of_80p:])
y_test = np.array(y[size_of_80p:])

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(X_test.shape))
############################################################################
# LSTM weights

# Weights for the input gate
generated_values1 = tf.truncated_normal([1,hidden_layers],stddev = 0.05)
generated_values2 = tf.truncated_normal([hidden_layers,hidden_layers], stddev = 0.05)
W_inputG = tf.Variable(generated_values1)       #  weights input gate
W_inputH = tf.Variable(generated_values2)       #  weights input hidden layer

