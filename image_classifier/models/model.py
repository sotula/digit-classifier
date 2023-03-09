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

