'''test_plots.py'''

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
from matplotlib import patches

def scatterPlot(figS, axS, time_series, edges, channels, calls):
	#format and save the scatterplot
	#first plot the scatterplot
	MAX_CHANS = 6
	FIG_COLS = 3
	FIG_ROWS = 2
	INCHS_PER_SUBPLOT = 10
	CORRECT_CAL = [0.0, 0.8, 0.4]
	NO_CAL = [0.0, 0.4, 0.8]
	MISS_CAL = [0.7, 0.0, 0.7]
	CHANNEL_TAG = 'CH_'
	COLORS = [[0, 0.4, 0.8],
			  [0, 0.8, 0.4],
			  [0.7, 1.0, 0.1],
			  [1.0, 0.7, 0.2],
			  [1.0, 0.2, 0.3],
			  [0.7, 0, 0.9]]

	fig1 = plt.figure(figsize=(FIG_ROWS*INCHES_PER_SUBPLOT, FIG_COLS*INCHES_PER_SUBPLOT))

	#let ground truth on binning be contrived calibrator
	if measured[row_i, col_i] == truth[row_i, col_i] and truth[row_i, col_i] >= 0:
		axS.plot([truth[row_i, col_i] + 0.75], [vals[channel]], marker = 'o', color = [0, 0.8, 0.4])
		plotColor = [0.0, 0.8, 0.4]
	elif measured[row_i, col_i] == -1 and truth[row_i, col_i] >= 0:
		axS.plot([truth[row_i, col_i] + 0.75], [vals[channel]], marker = '^', color = [0.0, 0.4, 0.8])
		plotColor = [0.0, 0.4, 0.8]
	elif truth[row_i, col_i] >= 0:
		axS.plot([truth[row_i, col_i] + 0.75], [vals[channel]], marker = 's', color = [0.7, 0.0, 0.7])
		plotColor = [0.7, 0.0, 0.7]

	axSc.grid('off')
	axSr.grid('off')
	axSc.patch.set_facecolor([1.0, 1.0, 1.0])
	#set x-ticks and x-tick labels for scatterplot
	axSc.set_xticks(np.arange(0.75, np.shape(bins3rdc)[0] + 0.75, 1))
	axSc.set_xticklabels(generateLabels(bins3rdc))
	axSc.set_xlabel('actual combinations', fontsize = 20, labelpad = 7.5)
	#set y-ticks and y-tick labels for scatterplot
	axSc.set_yticks(meansc[:])
	axSc_yticklabels = []
	for a in range(np.shape(meansc)[0]):
		axSc_yticklabels.append(str(a) + 'i ' + '%0.2f' % meansc[a])
	axSc.set_yticklabels(axSc_yticklabels)
	for tl in axSc.get_yticklabels():
		tl.set_color(RECT_COLORS[0])
	axSc.set_ylabel('contrived calibs', color = RECT_COLORS[0], fontsize = 20, rotation = 90, labelpad = 6.5)
	axSr.set_yticks(meansr[:])
	axSr.set_yticklabels(['%.2f' % a for a in meansr[:]])
	axSr.set_ylabel('real calibs', color = RECT_COLORS[1], fontsize = 20, rotation = 270, labelpad = 20)
	for tl in axSr.get_yticklabels():
		tl.set_color(RECT_COLORS[1])
		
	#draw the real bins 
	for l in range(np.shape(bins3rdr)[0]):
		x_lower = 0
		y_lower = bins3rdr[l, 0]
		rect_width = np.shape(bins3rdr)[0]
		rect_height = bins3rdr[l, 1] - bins3rdr[l, 0]
		axSr.add_patch(patches.Rectangle(
					   (x_lower, y_lower), #bottom left corner
					   rect_width, #width of rectangle
					   rect_height, #height of rectangle
					   facecolor = RECT_COLORS[l%2],
					   alpha = 0.1
					   ))
	#draw axis frame
	x_lower = 0 + 0.01
	y_lower = -1.1*np.abs(np.min(bins3rdc[0, 0], bins3rdr[0, 0])) + 0.01
	rect_width = np.shape(bins3rdc)[0] + 0.5 - 0.01
	rect_height = 1.1*np.max(bins3rdc[np.shape(bins3rdc)[0] - 1, 1], bins3rdr[np.shape(bins3rdr)[0] - 1, 1]) -0.01
	
	axSc.add_patch(patches.Rectangle(
				   (x_lower, y_lower), #bottom left corner
				   rect_width, #width of rectangle
				   rect_height, #height of rectangle
				   fill=False, #remove background
				   linewidth = 4.0
				   ))
	axSc.axis([x_lower - 0.01, rect_width + 0.01, y_lower - 0.01, rect_height + 0.01])
	axSr.axis([x_lower - 0.01, rect_width + 0.01, y_lower - 0.01, rect_height + 0.01])
	axSc.set_title(str(TEST_FILES[0] + RUN_LABEL + ALGO_USED + 'CH' + str(channel + 1) + 'scatter_3rd.png'), fontsize = 8, color = [0.3, 0.3, 0.3])
	figS.savefig(str('out_' + TEST_FILES[0] + RUN_LABEL + ALGO_USED + 'CH' + str(channel + 1) + '_scatter_3rd.png'), format = 'png', dpi = 150)

def plotWell(time_series, edges, channels, calls):
	'''plotting routine'''
	MAX_CHANS = 6
	FIG_COLS = 3
	FIG_ROWS = 2
	INCHS_PER_SUBPLOT = 10
	CORRECT_CAL = [0.0, 0.8, 0.4]
	NO_CAL = [0.0, 0.4, 0.8]
	MISS_CAL = [0.7, 0.0, 0.7]
	CHANNEL_TAG = 'CH_'
	COLORS = [[0, 0.4, 0.8],
			  [0, 0.8, 0.4],
			  [0.7, 1.0, 0.1],
			  [1.0, 0.7, 0.2],
			  [1.0, 0.2, 0.3],
			  [0.7, 0, 0.9]]

	fig1 = plt.figure(figsize=(FIG_ROWS*INCHES_PER_SUBPLOT, FIG_COLS*INCHES_PER_SUBPLOT))

	for i in range(np.shape(channels)[0]):
		ax = fig1.add_subplot(FIG_ROWS, FIG_COLS, i + 1)
		marker_shape = '.'
		#if it is a correct call
		if calls[i] == CORRECT_CAL:
			marker_shape = 'o'
		#if it is a no call
		elif calls[i] == NO_CAL:
			marker_shape = '^'
		#if it is a miss call
		if calls[i] == MISS_CAL:
			marker_shape = 's'

		ax.plot(time_series[i, :], color = calls[i], linewidth = 2.0)
		ax.plot(edges[i], time_series[i, edges[i]], marker = marker_shape, linewidth = 3.0, color = calls[i])
		ax.set_title(CHANNEL_TAG + str(channels[i] + 1), fontsize = 10.0)
		ax.set_xlabel('cycles', color = [0.4, 0.4, 0.4], fontsize = 12.0)
		ax.set_ylabel('intensity', color = COLORS[channels[i]], fontsize = 12.0)
		for tl in ax.get_yticklabels():
			tl.set_color(COLORS[channels[i]])

		for tl in ax.get_xticklabels():
			tl.set_color(COLORS[channels[i]])

	return fig1