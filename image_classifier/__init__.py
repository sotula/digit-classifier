import logging
from .models import *

logger = logging
logger.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def DigitClassifier(algorithm: str = RandomClassifier.model_type): #, 
	if algorithm == RandomClassifier.model_type:
		return RandomClassifier(logger=logger)