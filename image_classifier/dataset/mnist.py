from tensorflow.keras.datasets import mnist

def load_data():
	data = mnist.load_data()
	return data