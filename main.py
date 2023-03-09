import numpy as np
import logging
from image_classifier import DigitClassifier
from image_classifier.dataset import mnist


def main():
    model = DigitClassifier(algorithm='rand')  
    (trainX, trainy), (testX, testy) = mnist.load_data()
    print(testX.shape)

    model.load('./')
    print('prediction', model.predict(testX))


if __name__ == "__main__":
    main()