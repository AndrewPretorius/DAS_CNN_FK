import os
import pandas as pd
import numpy as np


### script to organise labels for CNN


# correct files and append to new label file

path2 = '/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/new_labels_27th/'
file_list2 = os.listdir(path2)

new_labels = pd.DataFrame(columns=['Filename','Label'])
labels_raw = pd.DataFrame(columns=['Filename','Label'])

for i in range(len(file_list2)):
    data2 = pd.read_csv(path2+file_list2[i],names=["Filename","Label"])
    labels_raw = labels_raw.append(data2, ignore_index = True)
    data2.Filename[0] = data2.Filename[0][:96] + str("%05.2f" % (np.float(data2.Filename[0][96:101]))) + ".npy"
    data2.Filename[1] = data2.Filename[1][:96] + str("%05.2f" % (np.float(data2.Filename[1][96:101]))) + ".npy"
    new_labels = new_labels.append(data2, ignore_index = True)
    
new_labels.to_csv('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/new_labels_22ndJune.csv')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


