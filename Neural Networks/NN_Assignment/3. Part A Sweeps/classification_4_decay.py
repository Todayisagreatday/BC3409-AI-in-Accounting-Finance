# Decay Optimization for 3 Layer NN
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, Input
from keras.regularizers import l2
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Hyperparameters
defaults=dict(
    epochs = 300,
    batch_size = 4,
    hidden_layer_1 = 15,
    lr = 0.01,
    decay=1e-6,
)
wandb.init(config=defaults, resume=True, name='Decay')
config = wandb.config
# Fix seed
seed = 10

np.random.seed(seed)
tf.random.set_seed(seed)

# Read data
data = np.genfromtxt(r'C:/Users/delim/Desktop/NN_Assignment/ctg_data_cleaned.csv', delimiter= ',')

# Preprocess data
X, y = data[1:, :21], data[1:,-1].astype(int)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Perform 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X_train, y_train):
    # Create model
    model = Sequential()
    model.add(Dense(config.hidden_layer_1, input_dim=21, activation='relu', kernel_regularizer=l2(config.decay)))
    model.add(Dense(3, activation='softmax'))
       
    # Compile model
    keras.optimizers.SGD(lr=config.lr)
    model.compile(optimizer='sgd',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train model
    model.fit(X_train[train], y_train[train],
            epochs=config.epochs,
            verbose = 2,
            batch_size=config.batch_size,
            validation_data=(X_train[test],y_train[test]),
            shuffle=True,
            callbacks=[WandbCallback()])

    # evaluate the model
    scores = model.evaluate(X_train[test], y_train[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
cv_acc = np.mean(cvscores)
cv_sd = np.std(cvscores)
wandb.log({'CV accuracy': cv_acc, 'CV Std Dev': cv_sd})