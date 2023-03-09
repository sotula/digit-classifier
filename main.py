import numpy as np
import logging
from image_classifier import DigitClassifier
from image_classifier.dataset import mnist


def main():
	# Choose algorithm
    # algorithm = 'rand'
    algorithm = 'cnn'
    # algorithm = 'rf'

    # Load model
    model = DigitClassifier(algorithm=algorithm) 

    # Load data
    (trainX, trainy), (testX, testy) = mnist.load_data()
    print(trainX.shape, trainy.shape)
    print(testX.shape, testy.shape)

    # Fit cnn model 
    # model.fit(trainX, trainy)
    # model.save(model_path='./models/cnn_model')

    # Make a prediction for img with size (1, 28, 28), but every model reshapes image to individual size
    print('Prediction for digit {}:'.format(testy[0]), model.predict(testX[:1, :, :]))


if __name__ == "__main__":
    main()