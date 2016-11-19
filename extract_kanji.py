#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py

SIZE_X = 64
SIZE_Y = 63

SAMPLE_WIDTH = 512
TOTAL_RECORDS = 51200
#TOTAL_RECORDS = 50560

DATA_FILE = "data/ETL8B/ETL8B2C1"




	
def console_output(tmp_str):
	output = ['']*SIZE_Y

	for row in range(SIZE_Y):
		for column in range(SIZE_X):
			output[row] = output[row] + tmp_str[row*SIZE_X+column]

	for row in range(SIZE_Y):
		print output[row]


def reader(counter):
	data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
	data = data[8:].encode('hex')
	
	tmp=[]
	for i in data:
		tmp.append(format(int(i,16),'04b'))

	tmp_str = ''.join(tmp)
	return tmp_str


def string_to_array(tmp_str):
	array = np.eye(SIZE_Y, SIZE_X).astype('uint8')

	for i in range(SIZE_Y):
		for j in range(SIZE_X):
			array[i][j] = 255*int(tmp_str[i*SIZE_X+j])

	return array


def progress_bar(counter):
	if counter in range(0, TOTAL_RECORDS, 1000):
		print counter
	if (counter+1) == TOTAL_RECORDS:
		print counter





sample = int(sys.argv[1])

file_handle = open(DATA_FILE, 'rb')
byte_buffer = file_handle.read( (TOTAL_RECORDS+1)*SAMPLE_WIDTH )
file_handle.close()




tmp_str = reader(sample)

tmp_array = string_to_array(tmp_str)



# Output of single Kanji

im = Image.fromarray(tmp_array, mode='L')

console_output(tmp_str)
im.show()
im.save('testEx.jpg')





