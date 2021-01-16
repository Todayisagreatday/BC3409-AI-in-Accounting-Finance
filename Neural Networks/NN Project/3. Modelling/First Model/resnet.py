import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

# Hyperparameters
defaults=dict(
    # Model 
    epochs = 35,
    batch_size = 128,
    fc_layer_1 = 2048,
    fc_layer_2 = 1024,
    learning_rate = 1e-3,
    optimizer = 'Adam'
)

wandb.init(config=defaults, resume=True, name='ResNet50 CPU', project='NN_Project_Test_Runs')
config = wandb.config

# Load dataset as dataframe
train_df = pd.read_csv("txt_files/gender_train.txt", sep=' ', names=['datadir', 'gender'])
test_df = pd.read_csv("txt_files/gender_test.txt", sep=' ', names=['datadir', 'gender'])
train_df['datadir'] = 'data/aligned/' + train_df['datadir'].astype(str)
test_df['datadir'] = 'data/aligned/' + test_df['datadir'].astype(str)

# Load images into keras image generator 
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_valid = ImageDataGenerator(rescale=1./255)
# For train generator
train_generator = datagen_train.flow_from_dataframe(
    dataframe = train_df,
    directory=None,
    x_col="datadir",
    y_col="gender",
    batch_size=config.batch_size,
    seed=7,
    class_mode='raw',
    target_size=(224,224))
# For test generator 
valid_generator = datagen_valid.flow_from_dataframe(
    dataframe = test_df,
    directory=None,
    x_col="datadir",
    y_col="gender",
    batch_size=config.batch_size,
    seed=7,
    class_mode='raw',
    target_size=(224,224))

# Define model
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(config.fc_layer_1, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(config.fc_layer_2, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable = False

# Compile model 
model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Fit and save model
model.fit(train_generator, epochs=config.epochs, validation_data=valid_generator, callbacks=[WandbCallback()])