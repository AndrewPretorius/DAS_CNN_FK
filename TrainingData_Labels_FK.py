from read_SEGY import read_SEGY
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy.signal import butter, sosfilt, hilbert
from numpy import savetxt
from obspy import UTCDateTime
from numpy.fft import fft2, fftshift, ifft2
from scipy.ndimage import median_filter
import pandas as pd

""" Script to import and convert passive data from segy to numpy arrays in 0.25s 
    windows for labelling as training data.
    
    Output is a csv file with each 0.25s window given an ID and a label 
    (0 = noise/negative, 1 = coherent arrivals/positive, 2 = ambiguous/needs correcting) """

# total channels n_ch and samples ns in file (should be n_ch=2688, and ns=12000 for 4KHz data / ns=3000 for 1KHz)
n_ch = 2688
ns = 120000

# useful channels
ch_min = 322
#ch_min = 372
#ch_min = 722
#ch_max = 822
ch_max = 1322

# define bandpass filter (for visualisation)
samp_rate = 4000
nyq = samp_rate/2
lowcut = 10
highcut = 150       # EDIT CUTOFFS
low = lowcut/nyq
high = highcut/nyq
order = 3
bandpass_filt = butter(order, [low,high], btype='band', analog=False, output='sos')

labels = pd.read_csv("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/new_labels_27th/labels_all_0.5.csv")


# specify location for training data folders to be created and the data stored
#folder_location = input("Please specify a full path for the directory to contain training data (2 new directories will be created): ")

folder_location = '/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows'
#os.mkdir(folder_location+"/true")
#os.mkdir(folder_location+"/false")
#os.mkdir(folder_location+"/other")
#os.mkdir(folder_location+"/labels")

# size and window_size of rolling window
window_size_s = 0.25 #window size in seconds
window_size = int(samp_rate*window_size_s) # window size in samples
downsample_factor = 4 # can downsample to save memory and time
no_of_files = 120 # number of files to use (each file = 30s, 120 files = 1 hour of data)

for j in range(392, len(labels)):
    
    new_labels = [] # initialise empty list of labels
    # Full path of segy data file
    file_id = labels.Filename[j][81:93]
    window_id = np.float(labels.Filename[j][95:99])
    filename = '/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/local_data/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    #filename = '/home/ee16a2p/Documents/PhD/DATA/Greenland/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    #filename = '/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    data_raw = read_SEGY(filename, n_ch, ns)   # read data
    data_raw = data_raw[ch_min:ch_max,:]           # trim data
    
    data_test = np.empty(data_raw.shape)
    ch_weighted = 600
    for i in range(ch_weighted):  # taper top 600 channels according to depth
        data_test[i,:] = data_raw[i,:]*((i/ch_weighted)**2)
    data_test[ch_weighted:,:] = data_raw[ch_weighted:,:]
    
    data = sosfilt(bandpass_filt, data_test)    # filter data for visualisation


    vm = np.percentile(data, 98)           # calculate percentile of array max values for visualisation
    #vm1 = np.percentile(envelope, 98)           # calculate percentile of array max values for visualisation
    window_number=0 # initialise window number
    # loop to move window over dataset
    for i in range(2):
        window_filt = data[:,int(window_id*samp_rate)+(i*window_size):int(window_id*samp_rate)+(i*window_size)+window_size]        #define each window from filtered data for visualisation
        
        #plt.figure(figsize=(9,9))  # for monitors
        plt.figure(1)
        plt.figure(figsize=(7,7))
        plt.title("F"+file_id+"_"+("%05.2f" % (window_id+(window_number*window_size_s)))+"S time domain")
        plt.imshow(window_filt, vmin=-vm, vmax=vm, extent=[0,window_size_s,1000,(ch_min-322)], aspect = 1/4000, cmap = 'gray')
        plt.ylabel('Depth (m)')
        plt.xlabel('Time (s)')

        plt.show()                #show window in console 
        
        decision = input("Does this window contain arrivals? (y/n) ") 
        
        # save window according to decision
        if decision == "y":
            #np.save(folder_location+"/true/F"+str(file_id)+"_"+str(window_number*window_size_s)+"S.npy", window) 
            new_labels.append(tuple(("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/samples/_true/F"+str(file_id)+"_W"+str("%05.2f" % (window_id+(window_number*window_size_s))),1))) # append labels with filename and 1 for positive
        elif decision == "n":
            #np.save(folder_location+"/false/F"+str(file_id)+"_"+str(window_number*window_size_s)+"S.npy", window)
            new_labels.append(tuple(("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/samples/false/F"+str(file_id)+"_W"+str("%05.2f" % (window_id+(window_number*window_size_s))),0))) # append labels filename and 0 for negative
        else:
            #np.save(folder_location+"/other/F"+str(file_id)+"_"+str(window_number*window_size_s)+"S.npy", window) # if accidentally press enter or unsure
            new_labels.append(tuple(("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/samples/other/F"+str(file_id)+"_W"+str("%05.2f" % (window_id+(window_number*window_size_s))),2))) # append labels filename
        window_number = window_number+1
        plt.close() #close plot so next window can be shownsavetxt(folder_location+'/labels/labels'+"_"+str(UTCDateTime(precision=0))+'.csv', new_labels, delimiter=',', fmt='%s')
    savetxt(folder_location+'/labels/new_labels_27th/labels_'+"F"+str(file_id)+"_"+str("%05.2f" % (window_id+(window_number*window_size_s)))+'.csv', new_labels, delimiter=',', fmt='%s')

#with open("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_windows/labels.txt", "wb") as fp:   #Pickling
#    pickle.dump(labels, fp)

""" TEST TRAIN SPLIT
import splitfolders
split_folders.ratio('Data', output="output", seed=1337, fixed=(100, 100), oversample=False, group_prefix=None)

DELETE EXCESS FILES FROM FALSE
from random import sample

files = os.listdir('path/to/your/folder')
for file in sample(files,11):
    os.remove(file)

"""








