from read_SEGY import read_SEGY
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy.signal import butter, sosfilt
from numpy import savetxt
from obspy import UTCDateTime
import pandas as pd
from numpy.fft import fft2, fftshift, fft
from scipy.ndimage import median_filter
from scipy import signal

""" Script to import and convert passive data from segy to numpy arrays in 0.25s 
    windows then FK transfromed for use as training data according to existing labels."""


labels = pd.read_csv("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/new_labels_22ndJune_corrected.csv")

old_location = "/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/local_data/"
FK_directory = "/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/samples/"


# total channels n_ch and samples ns in file (should be n_ch=2688, and ns=12000 for 4KHz data/ns=3000 for 1KHz)
n_ch = 2688
ns = 120000

samp_rate = 4000

# useful channels (remove last 44 channels)
ch_min = 322
ch_max = 1322

# size and window_size of rolling window
window_size_s = 0.25 #window size in seconds
window_size = int(samp_rate*window_size_s) # window size in samples

#for j in range(len(labels)):
for j in range(len(labels)):
    sample_id = labels.Filename[j]
    file_id = sample_id[82:94] # store file id string
    
    # Full path of segy data file
    filename = old_location + '/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    data = read_SEGY(filename, n_ch, ns)   # read data
    data = data[ch_min:ch_max,:]           # trim data
    
    window_number = float(sample_id[96:101]) # specify window within file
    window_raw = data[:,int(window_number*window_size):int(window_number*window_size)+window_size]
    
    ### add tapering to window
    
    taper_spatial = np.linspace(1,1000,1000)  # Apply tapering to each window
    taper1 = signal.tukey(2000, alpha=0.7)
    taper_time = signal.tukey(1000, alpha=0.1)
    taper_spatial[:900] = taper1[:900]
    taper_spatial[900:] = taper_time[900:]

    taper_2d_spatial = np.empty(window_raw.shape)
    taper_2d_time = np.empty(window_raw.shape)
    for i in range(1000):
        taper_2d_spatial[:,i] = taper_spatial
        taper_2d_time[i,:] = taper_time
    
    taper_2d = np.multiply(taper_2d_spatial,taper_2d_time)

    window_tapered = np.multiply(taper_2d,window_raw)
    
    #### 2D Fourier transform data, keeping both axes the same length
    NFFT=2**10;
    data_fk=fft2(np.transpose(window_tapered),(NFFT,NFFT+1));
    
    #### fftshift data
    data_shift=fftshift(data_fk);
    
    #### bandpass f 10Hz to 150 Hz, bandpass k between -0.04 and 0.04 m^-1
    wavn_neg = 472
    wavn_pos = 553
    freq_cut = 474
    data_fk_filt = np.empty((36,81))
    data_fk_complex = np.abs(data_shift[freq_cut:int((NFFT/2)+1)-3,wavn_neg:wavn_pos])
    
    ### save as new .npy file
    
    if labels.Label[j] == 0:
        np.save(FK_directory+"false/"+sample_id[81:101], data_fk_complex)
    elif labels.Label[j] == 1:
        np.save(FK_directory+"_true/"+sample_id[81:101], data_fk_complex)
        
    print("train set progress: "+str(j)+" out of "+str(len(labels))+" complete")
      