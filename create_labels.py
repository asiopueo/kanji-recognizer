#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py



SAMPLE_WIDTH = 512
TOTAL_RECORDS = 51200
#TOTAL_RECORDS = 50560

DATA_FILE = "data/ETL8B/ETL8B2C1"
OUTPUT_FILE = "c1_labels.hdf5"




def reader(counter):
	data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
	JIS_code = data[2:4].encode('hex')
	reading = data[4:8]

	JIS_code_ku = int(JIS_code[0:2], 16)-32
	JIS_code_ten = int(JIS_code[2:4], 16)-32
	
	return JIS_code_ku, JIS_code_ten


def progress_bar(counter):
	if counter in range(0, TOTAL_RECORDS, 1000):
		print counter
	if (counter+1) == TOTAL_RECORDS:
		print counter




#sample = int(sys.argv[1])

file_handle = open(DATA_FILE, 'rb')
byte_buffer = file_handle.read( (TOTAL_RECORDS+1)*SAMPLE_WIDTH )
file_handle.close()

t0 = time()
print "Creating hdf5-file..."

labels_training = np.ndarray( shape=(TOTAL_RECORDS*0.9, 2), dtype=int)
labels_validation = np.ndarray( shape=(TOTAL_RECORDS*0.1, 2), dtype=int)

counter_tr = 0
counter_va = 0

for counter in range(TOTAL_RECORDS):
	if (counter%10 == 0):
		labels_validation[counter_va] = reader(counter+1)
		counter_va += 1
	else:
		labels_training[counter_tr] = reader(counter+1)
		counter_tr += 1

	progress_bar(counter)


print "Creation time:" , round(time()-t0, 3), "seconds"





"""
	Define a target variable
	Export features to .hdf5
"""

file_handle = h5py.File(OUTPUT_FILE, "w")
file_handle.create_dataset('training set/labels', data=labels_training, compression='gzip')
file_handle.create_dataset('validation set/labels', data=labels_validation, compression='gzip')
file_handle.close()






# Output of single Kanji

#im = Image.fromarray(total_array, mode='L')

#console_output(total_array[sample])
#print buffer_array
#im.show()
#im.save('test.png')





