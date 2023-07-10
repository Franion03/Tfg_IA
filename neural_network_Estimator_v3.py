#https://towardsdatascience.com/hyperparameter-tuning-with-keras-tuner-283474fbfbe
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
#import multiprocessing as mp


class RegressionHyperModel_1Layer(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = keras.Sequential()

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        model.add(
            keras.layers.Dense(
                units=hp.Int('units_layer', min_value=4, max_value=12, step=2),
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
        
        model.add(layers.Dense(3, activation='softmax'))

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
        
        model.add(layers.Dense(3, activation='softmax'))

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
        
        model.add(layers.Dense(3, activation='softmax'))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.Recall(), 
                               keras.metrics.Precision()])

        return model
    
if __name__ == "__main__":
    model_type = 1
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

    data = pd.read_csv('.\TRAIN\\f1.csv')

    # Drop rows where 'size' column is not equal to only_size
    if (only_size is not None) and (only_size in data['size'].unique()):
        data = data[data['size'] == only_size]
    elif only_size is not None:
        raise ValueError(f'only_size variable ({only_size}) must be None or {sorted(data["size"].unique())}')
    #data.drop('size', axis=1, inplace=True)


    data = data.values

    if data_augmentation:
        ## INCREASING DATABASE
        num_columns = 14
        assert(num_columns == data.shape[1])
        array_roll = []
        for row in data:
            result = row[-1]
            size = row[-2]
            variances = row[0:num_columns-2]
            for i in range(1,num_columns-2):
                new_row = np.roll(variances, i)
                new_row = np.append(new_row, np.array([size, result]))
                array_roll.append(new_row)

        data = np.concatenate((data, np.array(array_roll)), axis=0)
        #pd.DataFrame(data).to_csv('/home/javier.ruiza/neural/manual_training_v4_augmented.csv')

    X = data[:, :-1]  # Todas las columnas de entrada [mdv01, mvd02, ..., mdv12, size]
    y = data[:, -1]   # Columna de salida [result]

    X_MDV = X[:, :-1]  # Columnas MDV [mdv01, mvd02, ..., mdv12]
    X_Sizes = X[:, -1] # Columna de tamaño [size]

    # Convert multi-label classification to binary classification
    y_encoded = keras.utils.to_categorical(y)

    # Declaramos la matriz final de entrada
    X_input = np.empty((X_MDV.shape[0], 0))

    if add_MDV:
        X_input = np.column_stack((X_input, X_MDV))

    if add_statistics:
        # Calcula el máximo, mínimo, promedio y varianza para las columnas MDV
        max_val = np.amax(X_MDV, axis=1)
        min_val = np.amin(X_MDV, axis=1)
        mean_val = np.mean(X_MDV, axis=1)
        variance_val = np.var(X_MDV, axis=1)

        X_input = np.column_stack((X_input, max_val, min_val, mean_val, variance_val))

    if add_sizes:
        X_input = np.column_stack((X_input, X_Sizes))

    if X_input.shape[1] == 0:
        raise ValueError(f'X_input has zero columns!')
    '''
    if standarization or normalization:
        if standarization:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_input = scaler.fit_transform(X_input)
    '''




    X_train, X_test, y_train, y_test = train_test_split(X_input, y_encoded, test_size=0.2, random_state=0)


    if standarization or normalization:
        if standarization:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    input_shape = (X_train.shape[1],)

    if model_type == 1:
        hypermodel_1 = RegressionHyperModel_1Layer(input_shape)
        tuner_hb1 = kt.Hyperband(
                        hypermodel_1.build,
                        objective=kt.Objective("val_recall", direction="max"),
                        max_epochs=1000,
                        factor=3,
                        directory='my_dir_model1',
                        project_name='Estimator_v3')
    elif model_type == 2:
        hypermodel_2 = RegressionHyperModel_2Layers(input_shape)
        tuner_hb2 = kt.Hyperband(
                        hypermodel_2,
                        objective=kt.Objective("val_recall", direction="max"),
                        max_epochs=1000,
                        factor=3,
                        directory='my_dir_model2',
                        project_name='Estimator_v3')
    elif model_type == 3:
        hypermodel_3 = RegressionHyperModel_3Layers(input_shape)
        tuner_hb3 = kt.Hyperband(
                        hypermodel_3,
                        objective=kt.Objective("val_recall", direction="max"),
                        max_epochs=1000,
                        factor=3,
                        directory='my_dir_model3',
                        project_name='Estimator_v3')
    else:
        print("Option not available")
        exit(-1)
                        
                        
    # Will stop training if the "val_loss" hasn't improved in 5 epochs.
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=10)


    #with mp.pool.ThreadPool(mp.cpu_count()) as p:
    #    p.map(lambda x: x.search(X_train, y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=0), [tuner_hb1,tuner_hb2,tuner_hb3])


    if model_type == 1:
        tuner_hb1.search(X_train, y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
        tuner_hb1.results_summary()
        best_hps1=tuner_hb1.get_best_hyperparameters(num_trials=1)[0]
    elif model_type == 2:
        tuner_hb2.search(X_train, y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
        tuner_hb2.results_summary()
        best_hps2=tuner_hb2.get_best_hyperparameters(num_trials=1)[0]
    elif model_type == 3:
        tuner_hb3.search(X_train, y_train, epochs=2000, validation_split=0.2, callbacks=[stop_early], verbose=1)
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
        history1 = model1.fit(X_train, y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
        eval_result1 = model1.evaluate(X_test, y_test)
        print("[test loss, test recall, test precision]:", eval_result1)
        #val_acc_per_epoch1 = history1.history['val_recall']
        #best_epoch1 = val_acc_per_epoch1.index(max(val_acc_per_epoch1)) + 1
        #print('Best epoch: %d' % (best_epoch1,))
        #hypermodel1 = tuner_hb1.hypermodel.build(best_hps1)
    if model_type == 2:
        model2 = tuner_hb2.hypermodel.build(best_hps2)
        history2 = model2.fit(X_train, y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
        eval_result2 = model2.evaluate(X_test, y_test)
        print("[test loss, test recall, test precision]:", eval_result2)
        #val_acc_per_epoch2 = history2.history['val_recall']
        #best_epoch2 = val_acc_per_epoch2.index(max(val_acc_per_epoch2)) + 1
        #print('Best epoch: %d' % (best_epoch2,))
        #hypermodel2 = tuner_hb2.hypermodel.build(best_hps2)
    if model_type == 3:
        model3 = tuner_hb3.hypermodel.build(best_hps3)
        history3 = model3.fit(X_train, y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
        eval_result3 = model3.evaluate(X_test, y_test)
        print("[test loss, test recall, test precision]:", eval_result3)
        #val_acc_per_epoch3 = history3.history['val_recall']
        #best_epoch3 = val_acc_per_epoch3.index(max(val_acc_per_epoch3)) + 1
        #print('Best epoch: %d' % (best_epoch3,))
        #hypermodel3 = tuner_hb3.hypermodel.build(best_hps3)

    '''
    # Retrain the model
    if model_type == 1:
        hypermodel1.fit(X_train, y_train, epochs=best_epoch1, validation_split=0.2)
        eval_result1 = hypermodel1.evaluate(X_test, y_test)
        print("[test loss, test recall, test precision]:", eval_result1)
    if model_type == 2:
        hypermodel2.fit(X_train, y_train, epochs=best_epoch2, validation_split=0.2)
        eval_result2 = hypermodel2.evaluate(X_test, y_test)
        print("[test loss, test recall, test precision]:", eval_result2)
    if model_type == 3:
        hypermodel3.fit(X_train, y_train, epochs=best_epoch3, validation_split=0.2)
        eval_result3 = hypermodel3.evaluate(X_test, y_test)
        print("[test loss, test recall, test precision]:", eval_result3)
    '''

    # Realizar predicciones en los datos de prueba
    if model_type == 1:
        y_pred = model1.predict(X_test)
    if model_type == 2:
        y_pred = model2.predict(X_test)
    if model_type == 3:
        y_pred = model3.predict(X_test)

    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(y_test, axis=1)

    print("f1_score: %0.5f" % (f1_score(y_test, y_pred, average="weighted")) )
    print("precision_score: %0.5f" % (precision_score(y_test, y_pred, average="weighted")) )
    print("recall_score: %0.5f" % (recall_score(y_test, y_pred, average="weighted")) )

    cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Plain','Edge','Texture']))
    disp.plot()
    plt.savefig('confusion_matrix_Layer' + str(model_type) + '.png')
    #cm_display.figure_ #.savefig('confusion_matrix.png')


    exit(0)

