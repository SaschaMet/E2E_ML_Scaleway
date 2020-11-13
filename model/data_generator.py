from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 64


def image_augmentation():
    idg = ImageDataGenerator(rescale=1 / 255.0,
                             horizontal_flip=True,
                             vertical_flip=False,
                             height_shift_range=0.1,
                             width_shift_range=0.1,
                             rotation_range=25,
                             shear_range=0.1,
                             zoom_range=0.15)
    return idg


def make_train_gen(df):
    idg = image_augmentation()
    train_gen = idg.flow_from_dataframe(dataframe=df,
                                        directory=None,
                                        x_col='path',
                                        y_col='Pneumonia',
                                        class_mode='raw',
                                        target_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE
                                        )
    return train_gen


def make_test_gen(valid_df):
    test_idg = ImageDataGenerator(rescale=1. / 255.0)
    test_gen = test_idg.flow_from_dataframe(dataframe=valid_df,
                                            directory=None,
                                            x_col='path',
                                            y_col='Pneumonia',
                                            class_mode='raw',
                                            shuffle=False,
                                            target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE)
    return test_gen


def create_train_data(train_df):
    train_gen = make_train_gen(train_df)
    return train_gen


def create_test_data(test_df):
    train_gen = make_test_gen(test_df)
    return train_gen
