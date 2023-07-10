import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperModel
import numpy as np
import glob
import os
from FileGenerator import BatchGenerator

class MLP(HyperModel):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input_shape = (input_size,)

    def build(self, hp):
        model = tf.keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int('units_layer', min_value=4, max_value=120, step=2),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=self.input_shape
            )        
        )
        
        model.add(
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        )

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Recall(), 
                               tf.keras.metrics.Precision()])

        return model
    
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
mlp = MLP(input_size=204, output_size=1)
tuner_hb1 = kt.BayesianOptimization(
                mlp,
                objective=kt.Objective("val_recall", direction="max"),
                max_trials=100,
                directory='perceptron',
                project_name='v1')
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  mode='min',
                                  patience=10,
                                  verbose=1,
                                  restore_best_weights=True)
]

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=10)
tuner_hb1.search(X_train, Y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
tuner_hb1.results_summary()
best_hps1=tuner_hb1.get_best_hyperparameters(num_trials=1)[0]
model1 = tuner_hb1.hypermodel.build(best_hps1)
history1 = model1.fit(X_train, Y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
eval_result1 = model1.evaluate(X_test, Y_test)
print("[test loss, test recall, test precision]:", eval_result1)
# mlp.train(X_train, Y_train)
# mlp.evaluate(X_test, Y_test)    
# predictions = mlp.predict(X_test)
# for test_input, prediction in zip(X_test, predictions):
#     print(f"Input: {test_input} => Prediction: {prediction}")
