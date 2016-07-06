'''
20160701_release_ccode_import.py
authors: juhwan yoo, aditya rajagopal, dominic yurk
changelog:
2016.06.30 | modified from ccode_import_example.py written by juhwan
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
import os
import numpy as np
import scipy as sp
import pickle
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

import csv
import xlrd
import datetime
import mimetypes

import ccode_lib_import as ccimp

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

def importData(file_name, data_dir, output_dir):
	# obtain the full directory name of the execution path.
	#currdir = os.getcwd();
	#currdir = currdir + '/'
	currdir = data_dir
	
	# specify input filename here.
	ex_filename = file_name;
	# specify filename of output.
	ex_save_filename = ex_filename[0:-4];   # set to same name as input
	                                        # data file.

	# configure options for import function here:
	'''
	'verbose'   -indicates level of reporting, used for debugging and does
  	          not currently output report even when set to true.
	'save2pkl'  -specifies if the data extracted from the input file should
	            be saved into pkl format.
	'data_dir'  -location of input data file
	'save_dir'  -location of directory in which you want the output pkl
 	           file saved.
	'save_name' -name of output save file, don't include extension in name
	'''

	myopts={'verbose':True, 'save2pkl':True, \
    	    'data_dir':currdir, 'save_dir':output_dir, 'save_name':ex_save_filename}


	#example function call. mydata is now a dictionary that contains within
	#it the data that was processed from qpcr machine data file ex_filename.
	mydata = ccimp.import_raw_data2pkl(ex_filename, myopts);

def import2Pickle(list_of_files, data_dir, output_dir):
	for i in list_of_files:
		importData(i, data_dir, output_dir)
