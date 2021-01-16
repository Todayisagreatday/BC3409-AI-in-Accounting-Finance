import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential, Input
from keras.regularizers import l2
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Hyperparameters
defaults=dict(
    epochs = 150,
    batch_size = 4,
    hidden_layer_1 = 20,
    hidden_layer_2 = 10,
    lr = 0.01,
    decay=0,
)
wandb.init(config=defaults, resume=True, name='Best 4L Network', project='Test Prediction')
config = wandb.config
# Fix seed
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

# read data
data = np.genfromtxt(r'C:/Users/delim/Desktop/NN_Assignment/Part A/ctg_data_cleaned.csv', delimiter= ',')

# X = predictors (21), y = target
X, y = data[1:, :21], data[1:,-1].astype(int)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Create model
model = Sequential()
model.add(Dense(config.hidden_layer_1, input_dim=21, activation='relu'))
model.add(Dense(config.hidden_layer_2, activation='relu', kernel_regularizer=l2(config.decay)))
model.add(Dense(3, activation='softmax'))
                
# Compile model
keras.optimizers.SGD(lr=config.lr)
model.compile(optimizer='sgd',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=config.epochs,
                    verbose = 2,
                    batch_size=config.batch_size,
                    shuffle=True,
                    callbacks=[WandbCallback()])