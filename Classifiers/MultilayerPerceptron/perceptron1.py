import tensorflow as tf
import numpy as np
import glob
import os
from FileGenerator import BatchGenerator

class MLP:
    def __init__(self, input_size, hidden_units, output_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(output_size, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, inputs, labels, epochs=100):
        self.model.fit(inputs, labels, epochs=epochs)

    def evaluate(self, inputs, labels):
        eval_result1 = self.model.evaluate(inputs, labels)
        print("[test loss, test recall, test precision]:", eval_result1)

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions.round()

path = r'.'
pathSplit = r'.\SPLIT'
trainPath = r'.\TRAIN'
testPath = r'.\TEST'
#definimos las listas de ficheros
trainFiles= glob.glob(os.path.join(trainPath , "f*.csv"))
testFiles= glob.glob(os.path.join(testPath , "f*.csv"))
batchsize=1000
traingenerator = BatchGenerator(trainFiles, batchsize, ';', validation=False)
testgenerator = BatchGenerator(testFiles, batchsize, ';', validation=False)


X_train,Y_train=next(traingenerator)
X_test,Y_test=next(testgenerator)

# Create an instance of the MLP class, train it, and make predictions
mlp = MLP(input_size=204, hidden_units=100, output_size=1)
mlp.train(X_train, Y_train)
mlp.evaluate(X_test, Y_test)
# predictions = mlp.predict(X_test)
# for test_input, prediction in zip(X_test, predictions):
#     print(f"Input: {test_input} => Prediction: {prediction}")
