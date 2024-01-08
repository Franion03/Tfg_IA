import datetime
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperModel
import numpy as np
import glob
import os
from FileGenerator import BatchGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class MLP(HyperModel):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input_shape = (input_size,)

    def build(self, hp):
        model = tf.keras.Sequential()

        num_layers = hp.Int('num_layers', min_value=1, max_value=8, default=3)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, default=0.25)
        
        model.add(tf.keras.layers.Input(shape=self.input_shape))

        for i in range(num_layers):
            units = hp.Int(f'units_{i}', min_value=self.output_size, max_value=self.input_size, step=2)
            model.add(tf.keras.layers.Dense(units, activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),))
            model.add(tf.keras.layers.Dropout(dropout_rate))  # Add a dropout layer after each dense layer

        model.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
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


# X_train,Y_train=next(traingenerator)
# X_test,Y_test=next(testgenerator)

# Create empty lists to store data
X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []

# Iterate over all batches in the training generator and append data to lists
for batch_num in range(80):
    X_batch, Y_batch = next(traingenerator)
    X_train_list.append(X_batch)
    Y_train_list.append(Y_batch)

# Iterate over all batches in the test generator and append data to lists
for batch_num in range(20):
    X_batch, Y_batch = next(testgenerator)
    X_test_list.append(X_batch)
    Y_test_list.append(Y_batch)

# Concatenate the data to get the final X_train, Y_train, X_test, and Y_test
X_train = np.concatenate(X_train_list)
Y_train = np.concatenate(Y_train_list)
X_test = np.concatenate(X_test_list)
Y_test = np.concatenate(Y_test_list)

# Convert Y_train and Y_test to categorical format (assuming it was one-hot encoded before)
Y_train = tf.keras.utils.to_categorical(Y_train)
Y_test = tf.keras.utils.to_categorical(Y_test)

# Create an instance of the MLP class, train it, and make predictions
mlp = MLP(input_size=len(X_train[0]), output_size=Y_train.shape[1])
tuner_hb1 = kt.BayesianOptimization(
                mlp,
                objective=kt.Objective("val_recall", direction="max"),
                max_trials=1000,
                directory='perceptron_categorical',
                project_name='v2')
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
best_model = tuner_hb1.get_best_models(num_models=1)[0]
# Save the entire model
best_model.save('perceptron.h5')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history1 = model1.fit(X_train, Y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
eval_result1 = model1.evaluate(X_test, Y_test)
best_model.save('perceptron1.h5')
print("[test loss, test recall, test precision]:", eval_result1)
y_pred = model1.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
Y_test=np.argmax(Y_test, axis=1)
print("f1_score: %0.5f" % (f1_score(Y_test, y_pred, average="weighted")) )
print("precision_score: %0.5f" % (precision_score(Y_test, y_pred, average="weighted")) )
print("recall_score: %0.5f" % (recall_score(Y_test, y_pred, average="weighted")) )

cm = confusion_matrix(Y_test, y_pred)
#print(cm)
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Plain','Edge','Texture']))
disp.plot()
plt.savefig('multilayerPerceptron' + '2 layer' + '.png')

