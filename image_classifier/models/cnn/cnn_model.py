# -*- coding: utf-8 -*-
"""
Random model digit classification
"""
import logging
import operator
import pickle
from pathlib import Path
from typing import Optional, List, Union, Dict

import numpy as np
from ..model import DigitClassificationInterface
import os
import tensorflow as tf


class CnnClassifier(DigitClassificationInterface):
    """
    Cnn digit classifier
    """
    model_type = 'cnn'

    def __init__(self,
                 model_path: Optional[Path] = None,
                 logger: logging.Logger = None):

        self.__logger = logger 
        self.__batch_size = 64
        self.__num_classes = 10
        self.__epochs = 2
        self.__model = None
        self.__reshape_size = [-1, 28, 28, 1]
        self.__input_shape = (28, 28, 1)
        self.__model_path = './models/cnn_model'
        if self.__model_path:
            self.load(self.__model_path)
        self.model_details = dict()


    def save(self,
             model_path: Path):
        """
        Save model
        :param Path model_path: path where to save

        """
        self.__logger.info(f'start saving cnn classifier model')
        self.__model.save(model_path)
        self.__logger.info(f'finish saving cnn classifier model')



    def load(self,
             model_path: Path):
        """
        Load model
        :param Path model_path: path to model

        """
        self.__logger.info(f'start loading cnn classifier model')
        self.__model = tf.keras.models.load_model(model_path)
        self.__logger.info(f'finish loading cnn classifier model')



        self.__logger.info(f'finish loading classifier model from {str(model_path)}')


    def __prepare_data(self, X: np.array, y: np.array = None) -> np.array:
        """
        Method for checking correct inputted data
        """

        if not len(X) or X is None:
            raise ValueError('Empty input data to predict')

        if not any([isinstance(X, np.ndarray), isinstance(X, list)]):
            raise TypeError('Expecting list or array; got %s' % type(X))

        X = np.array(X)
        X = np.reshape(X, self.__reshape_size)
        X = X / 255

        if y is not None:
            # y = tf.one_hot(y.astype(np.int32), depth=10)
            return X, y
        return X


    # define cnn model
    def __define_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=self.__input_shape ),
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.__num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model


    def fit(self,
            X: np.array,
            y: Union[List[int], np.array]):
        """
        Fits model on input train data

        Params:
        :param np.array X: train data
        :param Union[List[int], np.array] y: train data labels

        """
        self.__logger.info('...started preparing train data')
        X, y = self.__prepare_data(X, y)
        self.__logger.info('...finished preparing train data')


        print('X.shape', X.shape)
        self.__logger.info('...init cnn model')
        self.__model = self.__define_model()
        self.__logger.info('...finished cnn model')

        self.__logger.info('...started cnn model')
        history = self.__model.fit(X, y,
                    batch_size=self.__batch_size,
                    epochs=self.__epochs,
                    validation_split=0.1,
                    # scallbacks=[callbacks]
                    )
        self.__logger.info('...finished cnn model')



    def predict(self, X: np.array) -> np.array:
        """
        Predict classes

        Params

        :param np.array X: vectors or vector

        """

        X = self.__prepare_data(X)
        self.__logger.info('...reshaped X size: ' + str(X.shape))
        # Predict the values from the testing dataset
        y_pred = self.__model.predict(X)
        # Convert predictions classes to one hot vectors 
        y_pred_classes = np.argmax(y_pred,axis = 1) 

        return y_pred_classes[0]



    @property
    def model_details(self):
        """
        property model details
        """
        return super().model_details



    @model_details.setter
    def model_details(self, value):
        super(CnnClassifier, type(self)). \
            model_details.fset(self, value)

