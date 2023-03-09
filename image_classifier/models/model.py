"""
Module with base class for digit classifier models
"""
from abc import ABC, abstractmethod

class DigitClassificationInterface(ABC):
    """
    Base class for digit classifier models
    """
    __model_details: dict

    @property
    def model_type(self):
        """
        Property must be overridden in subclasses
        for build enum existed types
        """
        raise NotImplementedError


    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        predict
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        save
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        load
        """
        raise NotImplementedError

    @property
    def model_details(self):
        """
        :return:
        """
        return self.__model_details

    @model_details.setter
    def model_details(self, val):
        if isinstance(val, dict):
            self.__model_details = val
        else:
            raise TypeError(f'the `val` object must be dict, bytes or bytearray, not {type(val)}')
