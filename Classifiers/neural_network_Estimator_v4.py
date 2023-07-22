#https://towardsdatascience.com/hyperparameter-tuning-with-keras-tuner-283474fbfbe
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import keras_tuner as kt
from keras_tuner import HyperModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from FileGenerator import BatchGenerator
from FileGenerator import ValidationGenerator
from FileGenerator import SpectrumsCounter
from FileGenerator import FilesNumLines
from FileGenerator import FileNumLines
#import multiprocessing as mp

model_type = 3
# if len(sys.argv) > 1:
#     model_type = int(sys.argv[1])
# else:
#     print("Error, no argument passed")
#     exit(-1)

data_augmentation = False
only_size = None; # Puede ser [None, 4, 8, 16, 32, 64]

normalization = False
standarization = True

add_MDV = False
add_statistics = True
add_sizes = True

class RegressionHyperModel_1Layer(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        model.add(
            keras.layers.Dense(
                x=hp.Int('units_layer', min_value=4, max_value=12, step=2),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=input_shape
            )
        
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_1',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(layers.Dense(1, activation='softmax'))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.Recall(), 
                               keras.metrics.Precision()])

        return model

class RegressionHyperModel_2Layers(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        model.add(
            keras.layers.Dense(
                units=hp.Int('units_1', min_value=4, max_value=12, step=2),
                activation=hp.Choice(
                    'dense_activation_1',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=self.input_shape
            )
        
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_1',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(
            keras.layers.Dense(
                units=hp.Int('units_2', min_value=4, max_value=12, step=2),
                activation=hp.Choice(
                    'dense_activation_2',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
            )
        
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(layers.Dense(1, activation='softmax'))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.Recall(), 
                               keras.metrics.Precision()])

        return model

class RegressionHyperModel_3Layers(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        model.add(
            keras.layers.Dense(
                units=hp.Int('units_1', min_value=4, max_value=12, step=2),
                activation=hp.Choice(
                    'dense_activation_1',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=input_shape
            )
        
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_1',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(
            keras.layers.Dense(
                units=hp.Int('units_2', min_value=4, max_value=12, step=2),
                activation=hp.Choice(
                    'dense_activation_2',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
            )
        
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(
            keras.layers.Dense(
                units=hp.Int('units_3', min_value=4, max_value=12, step=2),
                activation=hp.Choice(
                    'dense_activation_3',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
            )
        
        )
        
        model.add(
            keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
        
        model.add(layers.Dense(1, activation='softmax'))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.Recall(), 
                               keras.metrics.Precision()])

        return model



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

data_augmentation = False
only_size = None; # Puede ser [None, 4, 8, 16, 32, 64]

normalization = False
standarization = True

add_MDV = False
add_statistics = True
add_sizes = True
model_type=1
# if standarization or normalization:
#     if standarization:
#         scaler = StandardScaler()
#     else:
#         scaler = MinMaxScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)


input_shape = (X_train.shape[1],)
print(input_shape)

if model_type == 1:
    hypermodel_1 = RegressionHyperModel_1Layer(input_shape)
    tuner_hb1 = kt.BayesianOptimization(
                    hypermodel_1,
                    objective=kt.Objective("val_recall", direction="max"),
                    max_trials=100,
                    directory='my_dir_model1',
                    project_name='Estimator_v4')
elif model_type == 2:
    hypermodel_2 = RegressionHyperModel_2Layers(input_shape)
    tuner_hb2 = kt.BayesianOptimization(
                    hypermodel_2,
                    objective=kt.Objective("val_recall", direction="max"),
                    max_trials=1000,
                    directory='my_dir_model2',
                    project_name='Estimator_v4')
elif model_type == 3:
    hypermodel_3 = RegressionHyperModel_3Layers(input_shape)
    tuner_hb3 = kt.BayesianOptimization(
                    hypermodel_3,
                    objective=kt.Objective("val_recall", direction="max"),
                    max_trials=2000,
                    directory='my_dir_model3',
                    project_name='Estimator_v4')
else:
    print("Option not available")
    exit(-1)
                    
                    
# Will stop training if the "val_loss" hasn't improved in 5 epochs.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=10)


#with mp.pool.ThreadPool(mp.cpu_count()) as p:
#    p.map(lambda x: x.search(X_train, Y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=0), [tuner_hb1,tuner_hb2,tuner_hb3])


if model_type == 1:
    tuner_hb1.search(X_train, Y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
    tuner_hb1.results_summary()
    best_hps1=tuner_hb1.get_best_hyperparameters(num_trials=1)[0]
elif model_type == 2:
    tuner_hb2.search(X_train, Y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
    tuner_hb2.results_summary()
    best_hps2=tuner_hb2.get_best_hyperparameters(num_trials=1)[0]
elif model_type == 3:
    tuner_hb3.search(X_train, Y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
    tuner_hb3.results_summary()
    best_hps3=tuner_hb3.get_best_hyperparameters(num_trials=1)[0]


# Get the optimal hyperparameters
'''
best_hps1=tuner_hb1.get_best_hyperparameters(num_trials=1)[0]
best_hps2=tuner_hb2.get_best_hyperparameters(num_trials=1)[0]
best_hps3=tuner_hb3.get_best_hyperparameters(num_trials=1)[0]
'''
'''
if model_type == 1:
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps1.get('units_A')}, and the optimal learning rate for the optimizer
    is {best_hps1.get('learning_rate')}.
    """)
elif model_type == 2:
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps2.get('units_A')}, and {best_hps2.get('units_B')} and the optimal learning rate for the optimizer
    is {best_hps2.get('learning_rate')}.
    """)
elif model_type == 3:
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps3.get('units_A')}, {best_hps3.get('units_B')} and {best_hps3.get('units_C')} and the optimal learning rate for the optimizer
    is {best_hps3.get('learning_rate')}.
    """)
'''

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss',
                                  mode='min',
                                  patience=10,
                                  verbose=1,
                                  restore_best_weights=True)
]
if model_type == 1:
    model1 = tuner_hb1.hypermodel.build(best_hps1)
    history1 = model1.fit(X_train, Y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
    eval_result1 = model1.evaluate(X_test, Y_test)
    print("[test loss, test recall, test precision]:", eval_result1)
    #val_acc_per_epoch1 = history1.history['val_recall']
    #best_epoch1 = val_acc_per_epoch1.index(max(val_acc_per_epoch1)) + 1
    #print('Best epoch: %d' % (best_epoch1,))
    #hypermodel1 = tuner_hb1.hypermodel.build(best_hps1)
if model_type == 2:
    model2 = tuner_hb2.hypermodel.build(best_hps2)
    history2 = model2.fit(X_train, Y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
    eval_result2 = model2.evaluate(X_test, Y_test)
    print("[test loss, test recall, test precision]:", eval_result2)
    #val_acc_per_epoch2 = history2.history['val_recall']
    #best_epoch2 = val_acc_per_epoch2.index(max(val_acc_per_epoch2)) + 1
    #print('Best epoch: %d' % (best_epoch2,))
    #hypermodel2 = tuner_hb2.hypermodel.build(best_hps2)
if model_type == 3:
    model3 = tuner_hb3.hypermodel.build(best_hps3)
    history3 = model3.fit(X_train, Y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
    eval_result3 = model3.evaluate(X_test, Y_test)
    print("[test loss, test recall, test precision]:", eval_result3)
    #val_acc_per_epoch3 = history3.history['val_recall']
    #best_epoch3 = val_acc_per_epoch3.index(max(val_acc_per_epoch3)) + 1
    #print('Best epoch: %d' % (best_epoch3,))
    #hypermodel3 = tuner_hb3.hypermodel.build(best_hps3)

'''
# Retrain the model
if model_type == 1:
    hypermodel1.fit(X_train, Y_train, epochs=best_epoch1, validation_split=0.2)
    eval_result1 = hypermodel1.evaluate(X_test, Y_test)
    print("[test loss, test recall, test precision]:", eval_result1)
if model_type == 2:
    hypermodel2.fit(X_train, Y_train, epochs=best_epoch2, validation_split=0.2)
    eval_result2 = hypermodel2.evaluate(X_test, Y_test)
    print("[test loss, test recall, test precision]:", eval_result2)
if model_type == 3:
    hypermodel3.fit(X_train, Y_train, epochs=best_epoch3, validation_split=0.2)
    eval_result3 = hypermodel3.evaluate(X_test, Y_test)
    print("[test loss, test recall, test precision]:", eval_result3)
'''

# Realizar predicciones en los datos de prueba
if model_type == 1:
    y_pred = model1.predict(X_test)
if model_type == 2:
    y_pred = model2.predict(X_test)
if model_type == 3:
    y_pred = model3.predict(X_test)

# y_pred=np.argmax(y_pred)
# Y_test=np.argmax(Y_test)

print("f1_score: %0.5f" % (f1_score(Y_test, y_pred, average="weighted")) )
print("precision_score: %0.5f" % (precision_score(Y_test, y_pred, average="weighted")) )
print("recall_score: %0.5f" % (recall_score(Y_test, y_pred, average="weighted")) )

cm = confusion_matrix(Y_test, y_pred)
#print(cm)
import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Plain','Edge','Texture']))
disp.plot()
plt.savefig('confusion_matrix_v4_Layer' + str(model_type) + '.png')
#cm_display.figure_ #.savefig('confusion_matrix.png')


exit(0)

