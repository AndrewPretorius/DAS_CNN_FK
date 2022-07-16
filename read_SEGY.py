"""
AP 26/11/2020
 
Function to read in SEGY formatted data as numpy array

CHECK FILE HEADER FOR INPUT VALUES n_ch, ns

Store glacier DAS data 30s 4kHz files: n_ch = 2688 ns = 120000

Requires filename of SEGY file, number of channels (n_ch) and number of
samples (ns) as input.

Trace data is assumed to be big-endian 32bit float binary format ('>f4')

Binary and textual headers assumed to be 3600 bytes

Trace headers assumed to be 60 bytes

This function doesn't read in header data, I suggest obspy if you need it
"""
import numpy as np
def read_SEGY(filename, n_ch, ns):

    # Open file and skip headers
    with open(filename, 'rb') as fid: 
        data_array = np.fromfile(fid, np.dtype('>f4'), offset=3600) 
    
    # reshape into n_ch traces of ns samples
    data = np.reshape(data_array,(ns+60,n_ch),order='F')
    
    # remove 60 byte trace headers
    data = data[60:,:]
    
    #transpose for intuitive geometry
    data = np.transpose(data)
    
    return data
