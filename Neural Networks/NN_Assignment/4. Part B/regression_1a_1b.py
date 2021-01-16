import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Hyperparameters
epochs = 200
batch_size = 8
hidden_layer_1 = 10
decay = 1e-3
lr=1e-3

seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

# Load and preprocess data
admit_data = np.genfromtxt(r'C:\Users\delim\Desktop\NN_Assignment\Part B\admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
Y_data = Y_data.reshape(Y_data.shape[0], 1)

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
history = model.fit(X_data, Y_data,
                    epochs=epochs,
                    verbose = 2,
                    batch_size=batch_size,
                    validation_split=0.3,
                    shuffle=True)

# Visualize metrics
# mse
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model Mean Sq Error')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('Part B/Plots/b_plot_mse_200.png')
plt.show()
# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('Part B/Plots/b_plot_loss_200.png')
plt.show()