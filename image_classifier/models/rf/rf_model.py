# -*- coding: utf-8 -*-
"""
Random Forest model for digit classification
"""
import logging
import operator
import pickle
from pathlib import Path
from typing import Optional, List, Union, Dict

import numpy as np
from ..model import DigitClassificationInterface
from sklearn.ensemble import RandomForestClassifier
import os


class RfClassifier(DigitClassificationInterface):
    """
    Random Forest digit classifier
    """
    model_type = 'rf'

    def __init__(self,
                 model_path: Optional[Path] = None,
                 logger: logging.Logger = None):

        self.__logger = logger 
        self.__model = None
        self.__reshape_size = [-1, 784]
        self.__model_path = './models/rf_model'
        self.__params ={
            'n_estimators':100
        }
        if self.__model_path:
            self.load(self.__model_path)
        self.model_details = dict()


    def save(self,
             model_path: Path):
        """
        Save model
        :param Path model_path: path where to save

        """
        self.__logger.info(f'start saving rf classifier model')
        with open(model_path, 'wb') as handle:
            pickle.dump(self.__model, handle)
        self.__logger.info(f'finish saving rf classifier model')



    def load(self,
             model_path: Path):
        """
        Load model
        :param Path model_path: path to model

        """
        self.__logger.info(f'start loading rf classifier model')
        with open(model_path, 'rb') as handle:
            self.__model = pickle.load(handle)
        self.__logger.info(f'finish loading rf classifier model')



    def __prepare_data(self, X: np.array) -> np.array:
        """
        Method for checking correct inputted data
        """

        if not len(X) or X is None:
            raise ValueError('Empty input data to predict')

        if not any([isinstance(X, np.ndarray), isinstance(X, list)]):
            raise TypeError('Expecting list or array; got %s' % type(X))

        X = np.array(X)
        X = np.reshape(X, self.__reshape_size)

        return X


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
        X = self.__prepare_data(X)
        self.__logger.info('...finished preparing train data')


        self.__logger.info('...init rf model')
        self.__model = RandomForestClassifier(**self.__params)
        self.__logger.info('...finished init rf model')

        self.__logger.info('...started rf model')
        self.__model.fit(X, y)
        self.__logger.info('...finished rf model')



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

        return y_pred[0]



