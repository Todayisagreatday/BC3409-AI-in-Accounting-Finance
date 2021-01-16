import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

# Hyperparameters
defaults=dict(
    epochs = 200,
    batch_size = 8,
    hidden_layer_1 = 10,
    decay = 1e-3,
    lr=1e-3,
    features=5
)
wandb.init(config=defaults, resume=True, name='5 Features', project='PLs Work')
config = wandb.config

# Fix seed
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

# Load and preprocess data
admit_data = np.genfromtxt(r'C:\Users\delim\Desktop\NN_Assignment\Part B\admission_predict_5f.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,:5], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)
idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]
X_data = (X_data- np.mean(X_data, axis=0))/ np.std(X_data, axis=0)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30, shuffle=True)

# Create model
model = Sequential()
model.add(Dense(config.hidden_layer_1, input_dim=5, activation='relu', kernel_regularizer=l2(config.decay)))
model.add(Dense(1, activation='linear'))

# Compile model
keras.optimizers.SGD(lr=config.lr)
model.compile(optimizer='sgd',
              loss=keras.losses.MeanSquaredError(),
              metrics=['mse'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=config.epochs,
                    verbose = 2,
                    batch_size=config.batch_size,
                    validation_data = (x_test, y_test),
                    callbacks=[WandbCallback()])

# Results
# 1/9 [==>...........................] - ETA: 0s - loss: 0.0124 - mse: 0.0059
# 9/9 [==============================] - 0s 597us/step - loss: 0.0116 - mse: 0.0051

# 1/4 [======>.......................] - ETA: 0s - loss: 0.0115 - mse: 0.0050
# 4/4 [==============================] - 0s 499us/step - loss: 0.0125 - mse: 0.0060