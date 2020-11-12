import os
from glob import glob
from pathlib import Path
from random import sample
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DIRECTORY_ROOT = os.path.abspath(Path(os.getcwd()))


def get_all_images():
    """Helper function to get the paths to all the images

    Returns:
        list: list of all image paths
    """
    all_image_paths = {os.path.basename(x): x for x in glob(
        os.path.join(DIRECTORY_ROOT + '/data', 'images*', '*.png'))}
    return all_image_paths


def prepare_dataset():
    """Helper function to get and prepare the dataset
        - Reduces the dataset to only the values for which we have images
        - Deletes not needed columnns
        - Adds a column for each finding
    Returns:
        pandas dataframe: prepared dataframe
    """
    df = pd.read_csv(DIRECTORY_ROOT + '/data/Data_Entry_2017.csv')
    df.columns = df.columns.str.replace(' ', '_')
    all_image_paths = get_all_images()
    image_paths = []

    for key in all_image_paths:
        image_paths.append(key)

    df = df[df['Image_Index'].isin(image_paths)]
    del df['View_Position']
    del df['OriginalImage[Width']
    del df['Height]']
    del df['OriginalImagePixelSpacing[x']
    del df['y]']
    del df['Unnamed:_11']
    all_labels = np.unique(
        list(chain(*df['Finding_Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            df[c_label] = df['Finding_Labels'].map(
                lambda finding: 1 if c_label in finding else 0)
    del df['Finding_Labels']

    # add a path column to the image
    df['path'] = df['Image_Index'].map(all_image_paths.get)

    return df


def create_splits(df, test_size, classToPredict):
    """Helper function to generate training and testing datasets

    Args:
        df (pandas dataframe): dataset from which the splits should be generated
        test_size (float): size of testing dataset
        classToPredict (string): name of the column in the dataset

    Returns:
        pandas dataframe: training and testing dataset
    """
    train_df, test_df = train_test_split(
        df,  test_size=test_size, stratify=df[classToPredict])

    print("Total Pneumonia cases: ", df['Pneumonia'].sum())
    print("train_df", train_df.shape)
    print("test_df", test_df.shape)

    # Check the distribution of pneumonia cases in the test and validation set
    print("train distribution before",
          train_df['Pneumonia'].sum()/len(train_df))
    print("test distribution before", test_df['Pneumonia'].sum()/len(test_df))

    # Get a balanced dataset (50/50) = equal amount of positive and negative cases in Training
    # so our model has enough cases to learn from
    p_inds = train_df[train_df.Pneumonia == 1].index.tolist()
    np_inds = train_df[train_df.Pneumonia == 0].index.tolist()

    np_sample = sample(np_inds, len(p_inds))
    train_df = train_df.loc[p_inds + np_sample]
    print("train distribution after",
          train_df['Pneumonia'].sum()/len(train_df))

    return train_df, test_df


def start():
    """Get, prepare and split the dataframe into test and training sets and save it locally
    """
    df = prepare_dataset()
    train_df, test_df = create_splits(df, 0.2, 'Pneumonia')
    print("lenght train df", len(train_df))
    print("lenght test df", len(test_df))
    print("Pneumonia cases train df", train_df['Pneumonia'].sum())
    print("Pneumonia cases test df", test_df['Pneumonia'].sum())
    # save train and testing data to /data
    train_df.to_csv(DIRECTORY_ROOT + '/data/training_set.csv')
    test_df.to_csv(DIRECTORY_ROOT + '/data/testing_set.csv')


if __name__ == '__main__':
    start()
