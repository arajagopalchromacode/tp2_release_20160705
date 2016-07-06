

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
TT_FILE_EXTENSION = '.csv'
CALIB_FILE_EXTENSION = '.csv'
CALIB_FILE_MASK1 = '_calibs'
TT_FILE_MASK1 = '_tt_ch'
DATA_DIR_LOC = '/data/'
SCRIPTS_DIR_LOC = '/scripts/'
CALIBS_DIR_LOC = '/calibs/'
TRUTH_TABLES_DIR_LOC = '/truth_tables/'
OUTPUTS_DIR_LOC = '/outputs/'
PICKLE_DIR_LOC = '/pkl/'
NUM_CHAN = 5
CORRECT_PLOT_COLOR = [0.0, 0.8, 0.4]
MISS_PLOT_COLOR = [0.0, 0.4, 0.8]
WRONG_PLOT_COLOR = [0.7, 0.0, 0.7]
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
calibs_dir = main_dir + CALIBS_DIR_LOC
truth_table_dir = main_dir + TRUTH_TABLES_DIR_LOC
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

#load truth table data for each pickle we want to analyze
truth_tables = []
for i in pkls_to_analyze:
	pkl_truth_tables = []
	#run for potentially every channel
	for j in range(NUM_CHAN):
		tt_name = i[:-4].replace(PICKLE_DIR_LOC,TRUTH_TABLES_DIR_LOC,1) + TT_FILE_MASK1 + str(j) + TT_FILE_EXTENSION
		if os.path.isfile(tt_name):
			pkl_truth_tables.append(np.genfromtxt(tt_name, delimiter = ','))

	#add the truth tables for each pkl to full list
	truth_tables.append(pkl_truth_tables)

#load calib data
calibs_full = []
files_in_calibs = os.listdir(calibs_dir)

for i in pkls_to_analyze:
	calib_name = i[:-4].replace(PICKLE_DIR_LOC,CALIBS_DIR_LOC,1) + CALIB_FILE_MASK1 + CALIB_FILE_EXTENSION
	if os.path.isfile(calib_name):
		calibs_full.append(np.genfromtxt(calib_name, delimiter = ','))
		
channels = []
for run_calibs in calibs_full:
	tmp = []
	for j in range(len(run_calibs)):
		for elem in run_calibs[j]:
			if elem != -1:
				tmp.append(j)
				break
	channels.append(tmp)

calibs = []
for elem in calibs_full:
	tmp = []
	for row in elem:
		for calib in row:
			if calib != -1:
				tmp.append(row)
				break
	calibs.append(tmp)
print('FINISHED IMPORT')
#print('GOING INTO ANALYSIS')
for i in range(len(pkls_to_analyze)):
	curves, analRes, edgeRes = ca.analyzePlate(pkls_to_analyze[i], calibs[i], channels[i], algo = False)
	print('ANALYZED PLATE '+str(i))
#	print('PLATE ANALYZED')
	colors = []
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
		call_table = np.zeros([2,3])
		bin_table_3rd = np.zeros([8,12])
		bin_table_NWS = np.zeros([8,12])
		vals_table = np.zeros([8,12])
		vals_table_ep = np.zeros([8,12])
		edges_table = np.zeros([8,12])
		edges_table_ep = np.zeros([8,12])
		means = analRes[j][1]
		
		bins3rd = ca.generateFracBins(means, frac = 1.0/3)
		binsNWS = ca.generateNWSBins(means)
		wellVals = analRes[j][0]
		for k in range(NUM_WELLS):
			row_k, col_k = ca.index2tuple(k)
			tt_bin = truth_tables[i][j][row_k][col_k]
			if tt_bin < 0:
				ts[k].append(-1)
				edges[k].append(-1)
				colors[k].append([0,0,0])
				continue
			ts[k].append(curves[j][k])
			edges[k].append(edgeRes[j][k])
			binRes = ca.binResult(wellVals[k], bins3rd)
			vals_table[row_k][col_k] = wellVals[k]
			vals_table_ep[row_k][col_k] = curves[j][k][len(curves[j][k])-1]
			edges_table[row_k][col_k] = edgeRes[j][k]
			edges_table_ep[row_k][col_k] = len(curves[j][k])-1
			bin_table_3rd[row_k][col_k] = binRes
			if binRes == tt_bin:
				colors[k].append(CORRECT_PLOT_COLOR)
				call_table[0][0] += 1
			elif binRes == -1:
				colors[k].append(MISS_PLOT_COLOR)
				call_table[0][1] += 1
			else:
				colors[k].append(WRONG_PLOT_COLOR)
				call_table[0][2] += 1
				
			binRes = ca.binResult(wellVals[k], binsNWS)
			bin_table_NWS[row_k][col_k] = binRes
			if binRes == tt_bin:
				call_table[1][0] += 1
			elif binRes == -1:
				call_table[1][1] += 1
			else:
				call_table[1][2] += 1
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_bin_table_3rd.csv',
											bin_table_3rd, delimiter = ',', fmt='%d')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_bin_table_NWS.csv',
											bin_table_NWS, delimiter = ',', fmt='%d')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_vals_table.csv',
											vals_table, delimiter = ',', fmt='%.3f')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_vals_table_ep.csv',
											vals_table_ep, delimiter = ',', fmt='%.3f')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_edges_table.csv',
											edges_table, delimiter = ',', fmt='%d')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_edges_table_ep.csv',
											edges_table_ep, delimiter = ',', fmt='%d')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_call_table.csv',
											call_table, delimiter = ',', fmt='%d')
		np.savetxt(outputs_dir+pkls_to_analyze[i][len(pkl_dir)+1:] + '_ch'+str(chans[j])+'_means_table.csv',
											means, delimiter = ',', fmt='%.3f')
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
