import numpy as np
import pandas as pd
from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data():
    (X_train,y_train),(X_test,y_test)=cifar10.load_data()
    X_train,X_test=X_train/255.0,X_test/255.0

    y_train_cat=to_categorical(y_train,10)
    y_test_cat=to_categorical(y_test,10)

    return X_train,y_train_cat,X_test,y_test_cat

def create_datagen(X_train):
    datagen=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
    datagen.fit(X_train)
    return datagen

