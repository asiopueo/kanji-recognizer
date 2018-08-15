#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py


	
def show_on_console(tmp_str):
	output = ['']*SIZE_Y

	for row in range(SIZE_Y):
		for column in range(SIZE_X):
			output[row] = output[row] + tmp_str[row*SIZE_X+column]

	for row in range(SIZE_Y):
		print output[row]


def show_kuten(counter):
	data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
	JIS_code = data[2:4].encode('hex')
	reading = data[4:8]

	JIS_code_ku = int(JIS_code[0:2], 16)-32
	JIS_code_ten = int(JIS_code[2:4], 16)-32
	
	return (JIS_code_ku, JIS_code_ten)



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
			array[i][j] = 255 * int(tmp_str[i*SIZE_X+j])

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
# 51200/160 = 320 different Kanji in dataset
# TOTAL_RECORDS = 50560
DATA_FILE = "data/ETL8B/ETL8B2C1"




with open(DATA_FILE, 'rb') as file_handle:
	byte_buffer = file_handle.read( (TOTAL_RECORDS+1)*SAMPLE_WIDTH )



index = int(sys.argv[1])
kanji_as_str = reader(index)
kanji_as_array = string_to_array(kanji_as_str)



# Output of a single Kanji:
show_on_console(kanji_as_str)
print "Kuten-index: (%i, %i)" % show_kuten(index)

im = Image.fromarray(kanji_as_array, mode='L')
im.show()
im.save('example_output.jpg')





