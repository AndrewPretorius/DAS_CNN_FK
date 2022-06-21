import numpy as np
#np.random.seed(10)
 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import random
import tensorflow as tf
from numpy.fft import fft2, fftshift
from scipy.ndimage import median_filter
from datetime import datetime
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold, KFold


def load_samples(csv_file_path):
    data = pd.read_csv(csv_file_path)
    file_names = list(data.iloc[:,0])
    # Get the labels present in the second column
    labels = list(data.iloc[:,1])
    samples=[]
    for samp,lab in zip(file_names,labels):
        samples.append([samp,lab])
    return samples

train_samples = load_samples('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_0.25all.csv')
#test_samples = load_samples('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/labels/labels_test.csv')
#val_samples = load_samples('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_0.25val.csv')

def np_batch_generator(samples, batch_size=16, shuffle_data=True):
    """
    #Yields the next training batch.
    #Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        #if shuffle_data == True:
        random.shuffle(samples)
        #else:
        #    continue

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
                # expand dimensions for input
                # window = np.expand_dims(window,axis=2)
                
                label = np_utils.to_categorical(label, 2)
                
                # Add sample to arrays
                X.append(window)
                Y.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X = np.array(X)
            Y = np.array(Y)
            
            # try reshape
            X = X.reshape(-1, 36, 81, 1)

            # batch normalisation between 0 and 1
            X = (X- np.min(X)) / (np.max(X) - np.min(X))
            #X = (X/np.max(np.absolute(X)))
            
            # make sure labels are integers
            #Y = Y.astype('uint8')

            # yield training batch            
            yield X, Y

train_generator = np_batch_generator(train_samples, batch_size=len(train_samples))#, shuffle_data=False)
X, Y = next(train_generator)

batch_size = 32

StepSize_T=len(train_samples)//batch_size
#StepSize_V=len(val_samples)//batch_size
#StepSize_test = len(test_samples)//batch_size

# Define some model architecture
num_filters = 8
filter_size = 15
pool_size = 2

# k-fold cross valication
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
cvscores = []
for train, test in kfold.split(X, Y):
    model = Sequential([
      Conv2D(filters=64, kernel_size=9, input_shape=(36,81, 1),padding='same', activation='relu'),
             #kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
      BatchNormalization(),
      MaxPooling2D(pool_size=pool_size),
      #Conv2D(16, 3, padding='same', activation='tanh'),
             #kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
      #BatchNormalization(),
      #MaxPooling2D(pool_size=pool_size),
    
      #Conv2D(16, 3, activation='tanh',padding='same'),
      #Conv2D(16, 9,activation='tanh',padding='same'),
      #BatchNormalization(),
      #MaxPooling2D(pool_size=pool_size),
      
      #Conv2D(8, 5,activation='tanh',padding='same'),
      #Conv2D(32, 5,activation='tanh',padding='same'),
      #MaxPooling2D(pool_size=pool_size),
      
      #Conv2D(64, filter_size,activation='tanh',padding='same'),
      #Conv2D(64, filter_size,activation='tanh',padding='same'),
      #MaxPooling2D(pool_size=pool_size),
    
      Flatten(),
      Dense(64, activation='tanh'),#kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),  
      Dropout(0.5),
      Dense(2, activation='sigmoid'),
    ])

    # Compile model
    opt = SGD(lr=0.001, momentum=0, nesterov=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  #metrics=['accuracy'])
                  metrics=[BinaryAccuracy(),Precision(),Recall()])
    
    #stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    # Fit model on training data
    
    startTime = datetime.now()
    
    tf.autograph.set_verbosity(0)
    history = model.fit(X[train], Y[train],
                        batch_size=batch_size, 
                        epochs=50, 
                        verbose=1)
                        #steps_per_epoch=StepSize_T)
                        #validation_data=val_generator,
                        #validation_steps=StepSize_V,
                        #callbacks=stop_early)
    
    #print('Time taken to train model = ' + str(datetime.now() - startTime))
    
	# evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.3f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

"""
# Evaluate model on test data
score = model.evaluate(test_generator, batch_size=batch_size, steps = StepSize_test)

print("Model test loss = "+str(score[0]))
print("Model test binary accuracy = "+str(score[1]))
print("Model test precision = "+str(score[2]))
print("Model test recall = "+str(score[3]))

# plot
plt.figure(1)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = history.params['epochs']
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(2)
loss_train = history.history['binary_accuracy']
loss_val = history.history['val_binary_accuracy']
plt.plot(loss_train, 'g', label='Training accuracy')
plt.plot(loss_val, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""


# Save model
#model.save_weights('cnn100121.h5')
#model.load_weights('cnn100121.h5')
