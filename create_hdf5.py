#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py



class KutenDictionary(dict):
	def __setitem__(self, key, value):
		if key in self:
			del self[key]
		if value in self:
			del self[value]
		dict.__setitem__(self, key, value)
		dict.__setitem__(self, value, key)

	def __delitem__(self, key):
		dict.__delitem__(self, self[key])
		dict.__delitem__(self, key)

	def __len__(self):
		return dict.__len__(self) / 2





def reader4string(counter):
	data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
	data = data[8:].encode('hex')
	
	tmp=[]
	for i in data:
		tmp.append(format(int(i,16),'04b'))

	tmp_str = ''.join(tmp)
	return tmp_str


def reader(counter):
	data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
	JIS_code = data[2:4].encode('hex')
	reading = data[4:8]

	JIS_code_ku = int(JIS_code[0:2], 16)-32
	JIS_code_ten = int(JIS_code[2:4], 16)-32
	
	return (JIS_code_ku, JIS_code_ten)


def string_to_array(tmp_str):
	#array = np.eye(SIZE_Y, SIZE_X).astype('uint8')
	array = np.zeros(shape=(SIZE_Y+1, SIZE_X), dtype='uint8')

	for i in range(1, SIZE_Y):
		for j in range(SIZE_X):
			array[i][j] = int(tmp_str[i*SIZE_X+j])

	return array


def progress_bar(counter):
	if counter in range(0, TOTAL_RECORDS, 1000):
		print counter
	if (counter+1) == TOTAL_RECORDS:
		print counter




SIZE_X = 64
SIZE_Y = 63

SAMPLE_WIDTH = 512
TOTAL_RECORDS = 51200
#TOTAL_RECORDS = 50560
#320+320+316=956

DATA_FILE = "data/ETL8B/ETL8B2C1"
OUTPUT_FILE = "hdf5_data/testC1_normalized.hdf5"




with open(DATA_FILE, 'rb') as file_handle:
	byte_buffer = file_handle.read( (TOTAL_RECORDS+1) * SAMPLE_WIDTH )



print "Creating KuTen-dictionary..."
dictionary = KutenDictionary()

for counter in range(0,TOTAL_RECORDS,160):
	dictionary[counter/160] = reader(counter+1)
	


t0 = time()
print "Creating hdf5-file..."

# Using int() in order to avoid 'numpy deprecation warnings'
features_training = np.ndarray(shape=(int(TOTAL_RECORDS*0.9), SIZE_Y+1, SIZE_X))
labels_training = np.ndarray(shape=(int(TOTAL_RECORDS*0.9), 1))
features_validation = np.ndarray(shape=(int(TOTAL_RECORDS*0.1), SIZE_Y+1, SIZE_X))
labels_validation = np.ndarray(shape=(int(TOTAL_RECORDS*0.1), 1))





# Create datasets:
counter_tr = 0
counter_va = 0

for counter in range(TOTAL_RECORDS):
	tmp_str = reader4string(counter+1)
	if (counter%10 == 0):
		features_validation[counter_va] = string_to_array(tmp_str)
		labels_validation[counter_va] = dictionary[reader(counter+1)]
		counter_va += 1
	else:
		features_training[counter_tr] = string_to_array(tmp_str)
		labels_training[counter_tr] = dictionary[reader(counter+1)]
		counter_tr += 1

	progress_bar(counter)



print "Creation time:" , round(time()-t0, 3), "seconds"

# Additional padding was necvessary for Brainstorm for some reason...
"""
features_training = features_training.reshape( (1, int(TOTAL_RECORDS*0.9), SIZE_Y, SIZE_X, 1) )
labels_training = labels_training.reshape( (1, int(TOTAL_RECORDS*0.9), 1) )
features_validation = features_validation.reshape( (1, int(TOTAL_RECORDS*0.1), SIZE_Y, SIZE_X, 1) )
labels_validation = labels_validation.reshape( (1, int(TOTAL_RECORDS*0.1), 1) )
"""

features_training = features_training.reshape( (int(TOTAL_RECORDS*0.9), 1, SIZE_Y+1, SIZE_X) )
labels_training = labels_training.reshape( (int(TOTAL_RECORDS*0.9)) )
features_validation = features_validation.reshape( (int(TOTAL_RECORDS*0.1), 1, SIZE_Y+1, SIZE_X) )
labels_validation = labels_validation.reshape( (int(TOTAL_RECORDS*0.1)) )



"""
	Define a target variable
	Export features to .hdf5
"""

file_handle = h5py.File(OUTPUT_FILE, "w")
file_handle.create_dataset('training set/features', data=features_training, compression='gzip')
file_handle.create_dataset('training set/labels', data=labels_training, compression='gzip')
file_handle.create_dataset('validation set/features', data=features_validation, compression='gzip')
file_handle.create_dataset('validation set/labels', data=labels_validation, compression='gzip')
file_handle.close()



