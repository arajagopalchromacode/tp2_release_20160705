'''
example ccode_import file usage.
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
currdir = os.getcwd();
ex_filename = 'ccodeviaa7_arajagopal_20151110_run1.xls';

mydata = ccimp.import_raw_data2pkl(ex_filename);




