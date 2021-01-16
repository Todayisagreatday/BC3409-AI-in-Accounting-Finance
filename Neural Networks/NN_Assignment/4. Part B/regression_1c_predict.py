import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

seed = 10
epochs = 200
batch_size = 8
hidden_layer_1 = 10
decay = 1e-3
lr=1e-3

np.random.seed(seed)
tf.random.set_seed(seed)

# Load and preprocess data
admit_data = np.genfromtxt(r'C:\Users\delim\Desktop\NN_Assignment\Part B\admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)
idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]
X_data = (X_data- np.mean(X_data, axis=0))/ np.std(X_data, axis=0)
x_train, x_valid, y_train, y_valid = train_test_split(X_data, Y_data, test_size=0.30, shuffle=True)

# Create model
model = Sequential()
model.add(Dense(hidden_layer_1, input_dim=7, activation='relu', kernel_regularizer=l2(decay)))
model.add(Dense(1, activation='linear'))

# Compile model
keras.optimizers.SGD(lr=lr)
model.compile(optimizer='sgd',
              loss=keras.losses.MeanSquaredError(),
              metrics=['mse'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    verbose = 2,
                    batch_size=batch_size)
# Separate into predicted and actual values
predicted = model.predict(x=x_valid)
actual = y_valid

# Preprocess into neat df
conc_1 = np.vstack(actual)
conc_2 = np.vstack(predicted)
df = pd.DataFrame(conc_1)
df['predicted'] = pd.DataFrame(conc_2)
number = list(range(1, 121))
df['number'] = pd.Series(number)
df.rename(columns={0: "actual"})

# Save as csv file
df.to_csv('1c_plot.csv')