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


class RandomClassifier(DigitClassificationInterface):
    """
    Merchant xgboost classifier
    """
    model_type = 'rand'

    def __init__(self,
                 model_path: Optional[Path] = None,
                 logger: logging.Logger = None):

        self.__logger = logger 
        self.cropy = 10
        self.cropx = 10
        self.min_class_value = 1
        self.max_class_value = 10
        self.reshape_size = [28, 28, -1]
        self.model_details = dict()


    def save(self,
             model_path: Path):
        """
        Save model
        :param Path model_path: path where to save

        """
        self.__logger.info(f'finish saving Random classifier model')



    def load(self,
             model_path: Path):
        """
        Load model
        :param Path model_path: path to model

        """
        self.__logger.info(f'start loading Random classifier model')
        self.__logger.info(f'finish loading Random classifier model')



    def fit(self,
            X: np.array,
            y: Union[List[int], np.array]):

        self.__logger.info('...started fit Random model')
        self.__logger.info('...finished fit Random model')



    def __crop_center(self, img):
	    y,x,n = img.shape
	    startx = x//2-(self.cropx//2)
	    starty = y//2-(self.cropy//2)    
	    return img[starty:starty+self.cropy,startx:startx+self.cropx, :]



    # @staticmethod
    def __prepare_data(self, X: Union[List[List[float]], List[float], np.array]) -> np.array:
        """
        Method for checking correct inputted data
        """

        if not len(X) or X is None:
            raise ValueError('Empty input data to predict')

        if not any([isinstance(X, np.ndarray), isinstance(X, list)]):
            raise TypeError('Expecting list or array; got %s' % type(X))

        X = np.array(X)
        X = np.reshape(X, self.reshape_size)
        X = self.__crop_center(X)

        return X



    def predict(self, X: np.array, ) -> np.array:
        """
        Predict classes

        Params

        :param np.array X: vectors or vector

        """

        X = self.__prepare_data(X)
        n = X.shape[2]
        pred = np.random.randint(low=self.min_class_value, high=self.max_class_value, size=n, dtype=int)

        return pred



    @property
    def model_details(self):
        """
        property model details
        """
        return super().model_details



    @model_details.setter
    def model_details(self, value):
        super(RandomClassifier, type(self)). \
            model_details.fset(self, value)

