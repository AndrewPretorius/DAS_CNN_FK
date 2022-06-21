import numpy as np
#np.random.seed(10)
 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf
from keras_tuner.tuners import Hyperband, RandomSearch
import keras_tuner as kt
from keras_tuner.engine.hyperparameters import HyperParameters
import time
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def load_samples(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data = data[['Filename', 'Label']]
    file_names = list(data.iloc[:,0])
    # Get the labels present in the second column
    labels = list(data.iloc[:,1])
    samples=[]
    for samp,lab in zip(file_names,labels):
        samples.append([samp,lab])
    return samples

train_samples = load_samples('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_0.25train.csv')
#test_samples = load_samples('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/labels/labels_test.csv')
val_samples = load_samples('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_0.25val.csv')

def np_batch_generator(samples, batch_size=16): #shuffle_data=True):
    """
    #Yields the next training batch.
    #Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        #if shuffle_data == True
        random.shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X and Y lists for this batch
            X = []
            Y = []

            # For each sample
            for batch_sample in batch_samples:
                # Load FK domain window (X) and label (y)
                window_name = batch_sample[0]
                label = batch_sample[1]
                window =  np.load(window_name)
                
                #convert dtype
                window = window.astype('float32')
                
                label = np_utils.to_categorical(label, 2)
                
                # Add sample to arrays
                X.append(window)
                Y.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X = np.array(X)
            Y = np.array(Y)
            
            # reshape
            X = X.reshape(-1, 36, 81, 1)

            # batch normalisation between 0 and 1
            X = (X- np.min(X)) / (np.max(X) - np.min(X))

            # yield training batch            
            yield X, Y


train_generator = np_batch_generator(train_samples, batch_size=1678)#, shuffle_data=False)
val_generator = np_batch_generator(val_samples, batch_size=400)#, shuffle_data=True)
X_train, Y_train = next(train_generator)
X_val, Y_val = next(val_generator)

train_datagen = ImageDataGenerator(fill_mode='constant',brightness_range=[0.5,1.5], featurewise_center = True)
val_datagen = ImageDataGenerator(fill_mode='constant',brightness_range=[0.5,1.5], featurewise_center = True)

batch_size = 32

train_generator_aug = train_datagen.flow(X_train,Y_train,batch_size=batch_size,shuffle=True)
val_generator_aug = val_datagen.flow(X_val,Y_val,batch_size=batch_size,shuffle=True)

StepSize_T=len(train_samples)//batch_size
StepSize_V=len(val_samples)//batch_size
#StepSize_test = len(test_samples)//batch_size

""" 
FOR TUNING: no.of conv layers, no. of pooling layers, no. of filters, filter kernel sizes, 
padding?, activations, dense neurons, dropout, learning rate, momentum
    
"""
def build_model(hp):
    model = Sequential()
    
     # Tune the number of dense layers
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Conv2D(hp.Choice('num_filters_'+str(i+1),values=[8, 16, 32, 64],default=16), 
                     #hp.Choice('filt_size2',values=[3, 5, 7, 9],default=5), 
                     kernel_size=hp.Choice('kernel_'+str(i+1),values=[3, 5, 7, 9],default=5),
                     padding='same', 
                     activation=hp.Choice('activation',
                                          values=['relu', 'tanh'],
                                          default='tanh')))     
    
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=2))


    model.add(Flatten())
    
    model.add(Dense(units=hp.Choice('dense_units',values=[16,32,64],default=16), 
            activation=hp.Choice('activation',
                                 values=['relu', 'tanh'],
                                 default='tanh')))  
    
    model.add(Dropout(0.5))
    
    """
    model.add(Dropout(hp.Float('dropout_dense',
                    min_value=0.1,
                    max_value=0.5,
                    default=0.5,
                    step=0.1,)))
    """
    model.add(Dense(2, activation='sigmoid'))
    
    # Compile model
    opt = SGD(lr=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]),
              momentum=hp.Choice('momentum', values=[0.0, 0.5, 0.9]),
              nesterov=False)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[BinaryAccuracy(),Precision(),Recall()])
    
    return model

LOG_DIR = f"/home/ee16a2p/Documents/PhD/scripts/CNN/hyperparameter_tuning_logs/{int(time.time())}"

"""
tuner = kt.Hyperband(build_model,
                     objective='val_binary_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory=LOG_DIR,
                     project_name='DAS_FK_CNN')
"""
tuner = kt.BayesianOptimization(build_model,
                                objective='val_loss',
                                max_trials=300,
                                num_initial_points=2,
                                alpha=0.0001,
                                beta=2.6,
                                directory=LOG_DIR,
                                project_name='DAS_FK_CNN')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(train_generator_aug,
             batch_size=batch_size, 
             epochs=20, 
             verbose=1,
             steps_per_epoch=StepSize_T,
             validation_data=val_generator_aug,
             validation_steps=StepSize_V)

# optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
