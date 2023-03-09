import logging
from .models import *

logger = logging
logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def DigitClassifier(algorithm: str): 
	if algorithm == RandomClassifier.model_type:
		logger.info('RandomClassifier was initialized')
		return RandomClassifier(logger=logger)
	elif algorithm == CnnClassifier.model_type:
		logger.info('CnnClassifier was initialized')
		return CnnClassifier(logger=logger)
	elif algorithm == RfClassifier.model_type:
		logger.info('RfClassifier was initialized')
		return RfClassifier(logger=logger)