import logging
from .models import *

logger = logging
logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def DigitClassifier(algorithm: str): 
	print('RandomClassifier.model_type', RandomClassifier.model_type)
	print('CnnClassifier.model_type', CnnClassifier.model_type)
	if algorithm == RandomClassifier.model_type:
		return RandomClassifier(logger=logger)
	elif algorithm == CnnClassifier.model_type:
		return CnnClassifier(logger=logger)