# DAS_CNN_FK

Scripts to setup, prepocess data and run CNN on DAS data to identify seismicity 

CNN_FK_passive.py: script to run train CNN on fk domain training data

CNN_FK_passive_hptuning.py: hyperparameter tuning with fk domain training data

TrainingData_Labels_FK.py: Script to import, visualise and manually label in 0.25s windows. Outputs 1 CSV file (of window filenames and associated labels) per segy file.

organise_labels.py: Combines csv label files for each segy file into a single csv file of training labels

SaveDataAsFK.py: Loads segy files, time domain to fk conversion, saves as .npy files (according to training labels)

read_SEGY.py: Function to load segy files into numpy array
