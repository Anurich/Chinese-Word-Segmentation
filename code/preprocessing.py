#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:48:16 2019

@author: anupam
"""

import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from hanziconv import HanziConv




# function for preprocessing of the data and convert to desired format 
def preprocessedText(file_path=""):
	X_train = []
	
	Y_train = []
	if file_path == "":
		print("No File Exist")
	else:

		with open(file_path,'r') as fp:
			
			for line in fp:
				line  = HanziConv.toSimplified(line)

				splitted_data  = line.rstrip().split()
				splitted_data  = list(filter(None,splitted_data))

				save_word_dict = []				
				for word in splitted_data:
					count  = len(word)
					for i in range(len(word)):
						if count == 1:
							save_word_dict.append("S")
						else:
							if i== 0:
								save_word_dict.append("B")
							elif i > 0 and i < len(word)-1:
								save_word_dict.append("I")
							else:
								save_word_dict.append("E")
				
				
				X_train.append("".join(splitted_data))
				Y_train.append("".join(save_word_dict))


	return X_train,Y_train




#Function for writing the data into csv file 
def convertToDataFrame(x_train,y_train,type):

	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)

	if type =="cross_validation":
		
		writer = open("write_input_cv.txt",'a',encoding='utf8')
		x_train = "\n".join(x_train)
		writer.write(x_train)
		writer.close()

		writer = open("write_labels_cv.txt","a",encoding='utf8')
		y_train = "\n".join(y_train)
		writer.write(y_train)

		writer.close()

	elif type == 'train':
		writer = open("write_input.txt",'a',encoding='utf8')
		x_train = "\n".join(x_train)
		writer.write(x_train)
		writer.close()

		writer = open("write_labels.txt","a",encoding='utf8')
		y_train = "\n".join(y_train)
		writer.write(y_train)

		writer.close()

	elif type == 'test':
		writer = open("write_input_test.txt",'a',encoding='utf8')
		x_train = "\n".join(x_train)
		writer.write(x_train)
		writer.close()

		writer = open("write_labels_test.txt","a")
		y_train = "\n".join(y_train)
		writer.write(y_train)

		writer.close()
