import os
import pandas as pd
import numpy as np


### script to organise labels for CNN

# correct filenames by adding a sig fig
"""
path1 = '/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/new_negative/'
file_list1 = os.listdir(path1)

new_labels = pd.DataFrame(columns=['Filename','Label'])
labels_raw = pd.DataFrame(columns=['Filename','Label'])

for i in range(len(file_list1)):
    data = pd.read_csv(path1+file_list1[i],names=["Filename","Label"])
    labels_raw = labels_raw.append(data, ignore_index = True)
    old_id = np.float(data.Filename[1][96:100])
    new_id = old_id + 0.25
    data.Filename[0] = data.Filename[0][:96] + str("%05.2f" % (np.float(data.Filename[0][96:101]))) + ".npy"
    data.Filename[1] = data.Filename[1][:96] + str("%05.2f" % (new_id)) +".npy"
    new_labels = new_labels.append(data, ignore_index = True)
    
    
"""
# correct rest of files and append to new_labels

path2 = '/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/new_negative/'
file_list2 = os.listdir(path2)

new_labels = pd.DataFrame(columns=['Filename','Label'])
labels_raw = pd.DataFrame(columns=['Filename','Label'])

for i in range(len(file_list2)):
    data2 = pd.read_csv(path2+file_list2[i],names=["Filename","Label"])
    labels_raw = labels_raw.append(data2, ignore_index = True)
    data2.Filename[0] = data2.Filename[0][:96] + str("%05.2f" % (np.float(data2.Filename[0][96:101]))) + ".npy"
    #data2.Filename[1] = data2.Filename[1][:96] + str("%05.2f" % (np.float(data2.Filename[1][96:101]))) + ".npy"
    new_labels = new_labels.append(data2, ignore_index = True)
    
new_labels.to_csv('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_negative_new2.csv')

"""
### Convert negative label list to 0.25s

path = '/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_neg_uncorrected.csv'
data = pd.read_csv(path,names=["Filename","Label"])
for i in range(len(data)):
    data.Filename[i] = data.Filename[i][:95] + str("%05.2f" % (np.float(data.Filename[i][95:99]))) + ".npy"
    data.Filename[i] = data.Filename[i][:50] + "0.25" + data.Filename[i][53:]
    
data2 = data.copy()
for i in range(len(data)):
    old_id = np.float(data2.Filename[i][96:101])
    new_id = old_id + 0.25
    data2.Filename[i] = data2.Filename[i][:96] + str("%05.2f" % (new_id)) +".npy"
    #data2.Filename[i] = data2.Filename[i][:50] + "0.25" + data2.Filename[i][53:]
    
data = data.append(data2)

data.to_csv('/home/ee16a2p/Documents/PhD/DATA/passive_cnn_data/0.25s_FK_windows/labels/labels_negative_test.csv')
    
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


