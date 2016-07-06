

###IMPORT LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
from matplotlib import patches
import pickle
import os
import sys
import gc
from multiprocessing import Pool, cpu_count

###CONSTANT
IMPORT_FILE_EXTENSION = '.xls'
PKL_FILE_EXTENSION = '.pkl'
CALIB_FILE_EXTENSION = '.csv'
DATA_DIR_LOC = '/data/'
SCRIPTS_DIR_LOC = '/scripts/'
OUTPUTS_DIR_LOC = '/outputs/'
PICKLE_DIR_LOC = '/pkl/'
NUM_CHAN = 5
CORRECT_PLOT_COLOR = [0.0, 0.8, 0.4]
NUM_WELLS = 96
FIG_COLS = 2
FIG_ROWS = 3
INCHES_PER_SUBPLOT = 10
CPU_COUNT = cpu_count()

###PLOTTING FUNCTION FOR MULTIPROCESSING
def plotWells(inputs):
	inds, ts, edges, chans, colors, outputs = inputs
	fig = plt.figure(figsize=(FIG_ROWS*INCHES_PER_SUBPLOT, FIG_COLS*INCHES_PER_SUBPLOT))
	for j in inds:
		ca.plotWell(ts[j], edges[j], chans, colors[j], fig, outputs[j])
	plt.close(fig)

###MAIN
main_dir = os.getcwd()

#locate where we are
data_dir = main_dir + DATA_DIR_LOC
scripts_dir = main_dir + SCRIPTS_DIR_LOC
outputs_dir = main_dir + OUTPUTS_DIR_LOC
pkl_dir = main_dir + PICKLE_DIR_LOC

#import libraries
sys.path.insert(0, scripts_dir)
import release_ccode_import_20160701 as ci
import release_ccode_analyze_20160701 as ca

#figure out the list of files we need to interrogate
files_in_data = os.listdir(data_dir)
#list only the xls files
files_to_run = []

for i in files_in_data:
	if len(i) > len(IMPORT_FILE_EXTENSION) and i[-1*len(IMPORT_FILE_EXTENSION):] == IMPORT_FILE_EXTENSION:
		files_to_run.append(i) 

ci.import2Pickle(files_to_run, data_dir, pkl_dir)
print('IMPORTED PICKLES')

#figure out the list of pickles we need to analyze
files_in_pkl = os.listdir(pkl_dir)
#list only pkl files
pkls_to_analyze = []
for i in files_in_pkl:
	if len(i) > len(PKL_FILE_EXTENSION) and i[-1*len(PKL_FILE_EXTENSION):] == PKL_FILE_EXTENSION:
		pkls_to_analyze.append(pkl_dir+i)
		
channels = []
for i in range(len(pkls_to_analyze)):
	channels.append(range(NUM_CHAN))
	
#print('GOING INTO ANALYSIS')
for i in range(len(pkls_to_analyze)):
	curves, analRes, edgeRes = ca.analyzePlate(pkls_to_analyze[i], -1, channels[i], algo = False, dtree = False)
	print('ANALYZED PLATE '+str(i))
#	print('PLATE ANALYZED')
	chans = channels[i]
	if len(analRes[0]) == NUM_CHAN:
		chans = range(NUM_CHAN)
	ts = []
	edges = []
	colors = []
	outputs = []
	for j in range(NUM_WELLS):
		ts.append([])
		edges.append([])
		colors.append([])
		outputs.append(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_' + str(j) + '.png')
	for j in range(len(chans)):
		vals_table = np.zeros([8,12])
		edges_table = np.zeros([8,12])
		means = analRes[j][1]
		wellVals = analRes[j][0]
		
		for k in range(NUM_WELLS):
			row_k, col_k = ca.index2tuple(k)
			ts[k].append(curves[j][k])
			edges[k].append(edgeRes[j][k])
			vals_table[row_k][col_k] = wellVals[k]
			edges_table[row_k][col_k] = edgeRes[j][k]
			colors[k].append(CORRECT_PLOT_COLOR)

		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_vals_table_ep.csv',
											vals_table, delimiter = ',', fmt='%.3f')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_edges_table_ep.csv',
											edges_table, delimiter = ',', fmt='%d')
#	print('GOING TO PLOTS')
	inputs = []
	for j in range(CPU_COUNT):
		inds = range(j*NUM_WELLS/CPU_COUNT, (j+1)*NUM_WELLS/CPU_COUNT, 1)
		inputs.append([inds, ts, edges, chans, colors, outputs])
	pool = Pool()
	pool.map(plotWells, inputs)
	pool.close()
	pool.join()
	gc.collect()
	print('PLOTTED PLATE '+str(i))
