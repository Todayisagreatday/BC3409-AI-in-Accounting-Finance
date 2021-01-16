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
epochs = 1000
batch_size = 32
hidden_layer_1 = 10
hidden_layer_2 = 10
lr = 0.01
decay=1e-6

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
model.add(Dense(hidden_layer_1, input_dim=21, activation='relu'))
model.add(Dense(hidden_layer_2, activation='relu', kernel_regularizer=l2(decay)))
model.add(Dense(3, activation='softmax'))
                
# Compile model
keras.optimizers.SGD(lr=lr)
model.compile(optimizer='sgd',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=epochs,
                    verbose = 2,
                    batch_size=batch_size,
                    shuffle=True)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('Part A/Plots/5a_plot_acc.png')
plt.show()
# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('Part A/Plots/5a_plot_loss.png')
plt.show()
