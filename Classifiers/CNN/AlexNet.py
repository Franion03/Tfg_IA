import tensorflow as tf
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import keras_tuner as kt
from keras_tuner import HyperModel
import glob
import numpy as np
from FileGenerator import BatchGenerator
from functools import reduce
from sklearn.metrics import  f1_score, precision_score, recall_score,  confusion_matrix, ConfusionMatrixDisplay



__all__ = ['AlexNet', 'alexnet']




class AlexNet(HyperModel):
    def __init__(self, input_size, output_size, input_shape):
        self.input_size = input_size
        self.output_size = output_size
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential() 
  
        # 1st Convolutional Layer 
        model.add(tf.keras.Conv2D(filters = hp.Int('filters_Conv1', min_value=32, max_value=512, step=2), input_shape = self.input_shape,  
                    kernel_size = (hp.Int('kernel_size_x',
                                min_value=4,
                                max_value=256,
                                step=2), hp.Int('kernel_size_y',
                                min_value=4,
                                max_value=20,
                                step=2)), strides = ( hp.Int('strides_x',
                                min_value=2,
                                max_value=10,
                                step=2),  hp.Int('strides_y',
                                min_value=2,
                                max_value=10,
                                step=2)),  
                    padding = 'valid')) 
        model.add(tf.keras.Activation('relu')) 
        # Max-Pooling  
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(hp.Int('pool_size_x', min_value=1, max_value=4, step=1), hp.Int('pool_size_y', min_value=1, max_value=4, step=1)),
            strides=(hp.Int('strides_x', min_value=1, max_value=2, step=1), hp.Int('strides_y', min_value=1, max_value=2, step=1)),
            padding='valid'))
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # 2nd Convolutional Layer 
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int('filters_Conv2', min_value=32, max_value=512, step=2),
            kernel_size=hp.Int('kernel_size_Conv2', min_value=1, max_value=5, step=1),#TODO: change to tuple (x,y)
            strides=hp.Int('strides_Conv2', min_value=1, max_value=2, step=1),
            padding='valid'))
        model.add(tf.keras.Activation('relu')) 
        # Max-Pooling 
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(hp.Int('pool_size_x_pool2', min_value=1, max_value=4, step=1), hp.Int('pool_size_y_pool2', min_value=1, max_value=4, step=1)),
            strides=(hp.Int('strides_x_pool2', min_value=1, max_value=2, step=1), hp.Int('strides_y_pool2', min_value=1, max_value=2, step=1)),
            padding='valid'))
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # 3rd Convolutional Layer 
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int('filters_conv3', min_value=32, max_value=512, step=2),
            kernel_size=hp.Int('kernel_size_conv3', min_value=1, max_value=5, step=1), #TODO: change to tuple (x,y)
            strides=hp.Int('strides_conv3', min_value=1, max_value=2, step=1),
            padding='valid'))
        model.add(tf.keras.Activation('relu')) 
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # 4th Convolutional Layer 
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int('filters_conv4', min_value=32, max_value=512, step=2),
            kernel_size=hp.Int('kernel_size_conv4', min_value=1, max_value=5, step=1),#TODO: change to tuple (x,y)
            strides=hp.Int('strides_conv4', min_value=1, max_value=2, step=1),
            padding='valid'))
        model.add(tf.keras.Activation('relu')) 
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # 5th Convolutional Layer 
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int('filters_conv5', min_value=32, max_value=512, step=2),
            kernel_size=hp.Int('kernel_size_conv5', min_value=1, max_value=5, step=1),#TODO: change to tuple (x,y)
            strides=hp.Int('strides_conv5', min_value=1, max_value=2, step=1),
            padding='valid'))
        model.add(tf.keras.Activation('relu')) 
        # Max-Pooling 
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(hp.Int('pool_size_x_pool3', min_value=1, max_value=4, step=1), hp.Int('pool_size_y_pool3', min_value=1, max_value=4, step=1)),
            strides=(hp.Int('strides_x_pool3', min_value=1, max_value=2, step=1), hp.Int('strides_y_pool3', min_value=1, max_value=2, step=1)),
            padding='valid'))
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # Flattening 
        model.add(tf.keras.Flatten()) 
        
        # 1st Dense Layer 
        model.add(tf.keras.Dense(hp.Int('dense_layer1', min_value=1080, max_value=8180, step=512), input_shape = (reduce(lambda x, y: x*y, self.input_shape), )))  # TODO: change to real input multiplied by it own
        model.add(tf.keras.Activation('relu')) 
        # Add Dropout to prevent overfitting 
        model.add(tf.keras.Dropout(0.4)) 
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # 2nd Dense Layer 
        model.add(tf.keras.Dense(hp.Int('dense_layer2', min_value=1080, max_value=8180, step=512))) 
        model.add(tf.keras.Activation('relu')) 
        # Add Dropout 
        model.add(tf.keras.Dropout(0.4)) 
        # Batch Normalisation 
        model.add(tf.keras.BatchNormalization()) 
        
        # Output Softmax Layer 
        model.add(tf.keras.Dense(self.output_size)) 
        model.add(tf.keras.Activation('softmax')) 

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
mlp = AlexNet(input_size=len(X_train[0]), output_size=Y_train.shape[1])
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
best_model.save('alexNet.h5')
history1 = model1.fit(X_train, Y_train, epochs=1000, callbacks=callbacks, validation_split=0.2)
eval_result1 = model1.evaluate(X_test, Y_test)
best_model.save('AlexNet.h5')
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
plt.savefig('AlexNet' +'.png')