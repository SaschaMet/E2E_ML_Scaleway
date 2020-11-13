import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

# pylint: disable=import-error
from . import data_generator

EPOCHS = 30
LEARNING_RATE = 1e-4
DIRECTORY_ROOT = os.path.abspath(Path(os.getcwd()))


def load_datasets():
    test_df = pd.read_csv(DIRECTORY_ROOT + '/data/testing_set.csv')
    train_df = pd.read_csv(DIRECTORY_ROOT + '/data/training_set.csv')
    return train_df, test_df


def save_history(history):
    f = plt.figure()
    f.set_figwidth(15)

    f.add_subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label='val loss')
    plt.plot(history.history['loss'], label='train loss')
    plt.legend()
    plt.title("Modell Loss")

    f.add_subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.legend()
    plt.title("Modell Accuracy")

    plt.savefig(DIRECTORY_ROOT + '/model/history.png')


def load_pretrained_model(layer_of_interest="block5_pool"):

    model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = model.get_layer(layer_of_interest)
    vgg_model = Model(inputs=model.input, outputs=transfer_layer.output)

    for layer in vgg_model.layers[0:17]:
        layer.trainable = False

    return vgg_model


def build_model():

    model = Sequential()

    # add your pre-trained model,
    model.add(load_pretrained_model())

    # additional layers
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))
    return model


def train(model, train_df, test_df):
    epochs = EPOCHS
    optimizer = Adam(lr=LEARNING_RATE)
    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    test_gen = data_generator.create_test_data(test_df)
    train_gen = data_generator.create_train_data(train_df)

    testX, testY = test_gen.next()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    weight_path = DIRECTORY_ROOT + "/model/best.model.hdf5"

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto',
                                 save_weights_only=True)

    early = EarlyStopping(monitor='val_loss',
                          mode='auto',
                          patience=10)

    callbacks_list = [checkpoint, early]

    history = model.fit(train_gen,
                        validation_data=(testX, testY),
                        epochs=epochs,
                        callbacks=callbacks_list,
                        verbose=1)

    save_history(history)

    # save model architecture to a .json:
    model_json = model.to_json()
    with open(DIRECTORY_ROOT + "/model/my_model.json", "w") as json_file:
        json_file.write(model_json)


def start():
    train_df, test_df = load_datasets()
    model = build_model()
    # free up memory
    tf.keras.backend.clear_session()
    print("Model Summary", model.summary())
    train(model, train_df, test_df)


start()
