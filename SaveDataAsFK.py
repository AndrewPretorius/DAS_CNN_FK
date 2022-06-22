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

""" Script to import and convert passive data from segy to numpy arrays in 0.5s 
    windows then FK transfromed for use as training data according to existing labels."""

# VALIDATION SET

labels = pd.read_csv("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/labels/labels_val.csv")
labels.Filename[570:] = "_" + labels.Filename[570:].astype(str)

old_location = "/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/"
FK_directory = "/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/samples/"


# total channels n_ch and samples ns in file (should be n_ch=2688, and ns=12000 for 4KHz data/ns=3000 for 1KHz)
n_ch = 2688
ns = 120000

samp_rate = 4000

# useful channels (remove first 30 and last 44 channels)
ch_min = 322
ch_max = 1322

# size and window_size of rolling window
window_size_s = 0.5 #window size in seconds
window_size = int(samp_rate*window_size_s) # window size in samples

no_of_files = 120 # number of files to use (each file = 30s, 120 files = 1 hour of data)
#t = datetime.datetime(2019,7,7,4,52,52)  # set as date and time of first file to be used, check folder for this
"""
#for j in range(len(labels)):
for j in range(4546, len(labels)):
    sample_id = labels.Filename[j]
    file_id = sample_id[81:93] # store file id string
    
    # Full path of segy data file
    filename = '/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    data = read_SEGY(filename, n_ch, ns)   # read data
    data = data[ch_min:ch_max,:]           # trim data
    
    window_number = float(sample_id[95:99]) # specify window within file
    window_raw = data[:,int(window_number*window_size):int(window_number*window_size)+window_size]
    
   ### add tapering to window
    
    taper_spatial = np.linspace(1,1000,1000)  # Apply tapering to each window
    taper1 = signal.tukey(2000, alpha=0.7)
    taper_time = signal.tukey(2000, alpha=0.05)
    taper_spatial[:900] = taper1[:900]
    taper_spatial[900:] = taper_time[1900:]

    taper_2d_spatial = np.empty(window_raw.shape)
    taper_2d_time = np.empty(window_raw.shape)
    for i in range(2000):
        taper_2d_spatial[:,i] = taper_spatial
    for i in range(1000):
        taper_2d_time[i,:] = taper_time
    
    taper_2d = np.multiply(taper_2d_spatial,taper_2d_time)

    window_tapered = np.multiply(taper_2d,window_raw)
    
    #### 2D Fourier transform data, keeping both axes the same length
    NFFT=2**11;
    data_fk=fft2(np.transpose(window_tapered),(NFFT,NFFT+1));
    
    #### fftshift data
    data_shift=fftshift(data_fk);
    
    #### lowpass f 150 Hz, bandpass k between -0.04 and 0.04 m^-1

    wavn_neg = 942
    wavn_pos = 1105
    #freq_cut = 897
    freq_cut = 948
    data_fk_filt = np.abs(data_shift[freq_cut:int(NFFT/2+1)-6,wavn_neg:wavn_pos])
    
    #vm = np.percentile(data_fk_filt, 98)
    #plt.imshow(data_fk_filt, vmin = 0, vmax = vm)
    
    ### save as new .npy file
    
    if labels.Label[j] == 0:
        np.save(FK_directory+"/false/"+sample_id[80:99], data_fk_filt)
    elif labels.Label[j] == 1:
        np.save(FK_directory+"/true/"+sample_id[80:99], data_fk_filt)
        
    print("progress: "+str(j)+" out of "+str(len(labels))+" complete")
        
"""
# TRAINING SET

labels = pd.read_csv("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/labels/labels_train.csv")
labels.Filename[4547:] = "_" + labels.Filename[4547:].astype(str)

#old_location = "/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/"
#FK_directory = "/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/samples/filt_stack/"

samp_rate = 4000

# useful channels (remove first 30 and last 44 channels)
ch_min = 322
ch_max = 1322

# total channels n_ch and samples ns in file (should be n_ch=2688, and ns=12000 for 4KHz data/ns=3000 for 1KHz)
n_ch = 2688
ns = 120000

samp_rate = 4000

# useful channels
#ch_min = 322
#ch_max = 1366

# size and window_size of rolling window
window_size_s = 0.25 #window size in seconds
window_size = int(samp_rate*window_size_s) # window size in samples

no_of_files = 120 # number of files to use (each file = 30s, 120 files = 1 hour of data)
t = datetime.datetime(2019,7,7,4,52,52)  # set as date and time of first file to be used, check folder for this

#for j in range(len(labels)):
for j in range(4547, len(labels)):
    sample_id = labels.Filename[j]
    file_id = sample_id[81:93] # store file id string
    
    # Full path of segy data file
    filename = '/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    data = read_SEGY(filename, n_ch, ns)   # read data
    data = data[ch_min:ch_max,:]           # trim data
    
    window_number = float(sample_id[95:99]) # specify window within file
    window_raw = data[:,int(window_number*window_size):int(window_number*window_size)+window_size]
    
   ### add tapering to window
    
    taper_spatial = np.linspace(1,1000,1000)  # Apply tapering to each window
    taper1 = signal.tukey(2000, alpha=0.7)
    taper_time = signal.tukey(2000, alpha=0.05)
    taper_spatial[:900] = taper1[:900]
    taper_spatial[900:] = taper_time[1900:]

    taper_2d_spatial = np.empty(window_raw.shape)
    taper_2d_time = np.empty(window_raw.shape)
    for i in range(2000):
        taper_2d_spatial[:,i] = taper_spatial
    for i in range(1000):
        taper_2d_time[i,:] = taper_time
    
    taper_2d = np.multiply(taper_2d_spatial,taper_2d_time)

    window_tapered = np.multiply(taper_2d,window_raw)
    
    #### 2D Fourier transform data, keeping both axes the same length
    NFFT=2**11;
    data_fk=fft2(np.transpose(window_tapered),(NFFT,NFFT+1));
    
    #### fftshift data
    data_shift=fftshift(data_fk);
    
    #### lowpass f 150 Hz, bandpass k between -0.04 and 0.04 m^-1
    wavn_neg = 942
    wavn_pos = 1105
    #freq_cut = 897
    freq_cut = 948
    data_fk_filt = np.abs(data_shift[freq_cut:int(NFFT/2+1)-6,wavn_neg:wavn_pos])
    
    #vm = np.percentile(data_fk_filt, 98)
    #plt.imshow(data_fk_filt, vmin = 0, vmax = vm)
    
    ### save as new .npy file
    
    if labels.Label[j] == 0:
        np.save(FK_directory+"/false/"+sample_id[80:99], data_fk_filt)
    elif labels.Label[j] == 1:
        np.save(FK_directory+"/true/"+sample_id[80:99], data_fk_filt)
        
    print("progress: "+str(j)+" out of "+str(len(labels))+" complete")

"""        
# TEST SET

labels = pd.read_csv("/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/labels/labels_test.csv")
labels.Filename[569:] = "_" + labels.Filename[569:].astype(str)

#old_location = "/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/"
#FK_directory = "/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.5s_FK_windows/samples/filt_stack/"


# total channels n_ch and samples ns in file (should be n_ch=2688, and ns=12000 for 4KHz data/ns=3000 for 1KHz)
n_ch = 2688
ns = 120000

samp_rate = 4000

# useful channels (remove first 30 and last 44 channels)
#ch_min = 352
#ch_max = 1322

# size and window_size of rolling window
window_size_s = 0.5 #window size in seconds
window_size = int(samp_rate*window_size_s) # window size in samples

no_of_files = 120 # number of files to use (each file = 30s, 120 files = 1 hour of data)
t = datetime.datetime(2019,7,7,4,52,52)  # set as date and time of first file to be used, check folder for this

#for j in range(len(labels)):
for j in range(len(labels)):
    sample_id = labels.Filename[j]
    file_id = sample_id[81:93] # store file id string
    
    # Full path of segy data file
    filename = '/run/media/ee16a2p/Seagate Expansion Drive/Greenland Part 1/4kHz/Greenland_iDAS15040_ContinuousAQ_'+file_id+'.sgy'
    data = read_SEGY(filename, n_ch, ns)   # read data
    data = data[ch_min:ch_max,:]           # trim data
    
    window_number = float(sample_id[95:99]) # specify window within file
    data_raw = data[:,int(window_number*window_size):int(window_number*window_size)+window_size]
    
    window = np.transpose(data_raw)
    
    # median filter
    #data_medfilt = median_filter(data_raw, 7)
    
    # stack data
    #data_stacked = np.empty(data_raw[:,::2].shape)
    #for i in range(len(data_stacked[0,:])):
    #    data_stacked[:,i] = data_medfilt[:,int(i*2)] + data_medfilt[:,int((i*2)+1)]# + data_medfilt[:,int((i*4)+2)] + data_medfilt[:,int((i*4)+3)]
    

    
    #### 2D Fourier transform data, keeping both axes the same length
    NFFT=2**11;
    data_fk=fft2(window,(NFFT,NFFT+1));
    
    #### fftshift data
    data_shift=fftshift(data_fk);
    
    #### lowpass f 150 Hz, bandpass k between -0.04 and 0.04 m^-1
    wavn_neg = 942
    wavn_pos = 1105
    #freq_cut = 897
    freq_cut = 948
    data_fk_filt = np.abs(data_shift[freq_cut:int(NFFT/2+1),wavn_neg:wavn_pos])
    
    #vm = np.percentile(data_fk_filt, 98)
    #plt.imshow(data_fk_filt, vmin = 0, vmax = vm)
    
    ### save as new .npy file
    
    if labels.Label[j] == 0:
        np.save(FK_directory+"/false/"+sample_id[80:99], data_fk_filt)
    elif labels.Label[j] == 1:
        np.save(FK_directory+"/true/"+sample_id[80:99], data_fk_filt)
        
    print("progress: "+str(j)+" out of "+str(len(labels))+" complete")
"""