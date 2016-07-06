'''
Description:    This file contains functions for curve shapping, curve
                interrogation, curve plotting, bin generation, and 
                results generation.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import patches
import pickle
import scipy.stats as st
#import gc
from scipy.signal import argrelextrema
#import threading
from multiprocessing import Pool

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Description:    This section contains functions to assist in 
                normalization and cross-talk cancellation.

Authors:		Juhwan Yoo
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NTC_thresh = 0.05;

# cross-talk matrix definition extracted by Luca
xtalk_mat_20160504_v0 = np.matrix(\
    [[  1.0,    0.0357,     0.0008,     0.0402,     0.0002  ],\
    [   0.0661, 1,          0.0921,     0.0038,     0       ],\
    [   0.0005, 0.375,      1,          0.011,      0.0001  ],\
    [   0,      0.0038,     0.4861,     1,          0.0075  ],\
    [   0.0001, -0.0003,    0.0002,     0.0019,     1       ]]);

xtalk_mat_20160504_v1 = np.matrix(\
    [[  1.0,    0.0357,     0,          0.0402,     0       ],\
    [   0.0661, 1,          0.0921,     0.0038,     0       ],\
    [   0,      0.375,      1,          0.011,      0       ],\
    [   0,      0.0038,     0.4861,     1,          0.0075  ],\
    [   0,      0,          0,          0.0019,     1       ]]);

# cross-talk inverse matrices
xtalk_mat_20160504_inv_v0 = np.linalg.inv(xtalk_mat_20160504_v0);
xtalk_mat_20160504_inv_v1 = np.linalg.inv(xtalk_mat_20160504_v1);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Description:    Returns submatrix specified by 0-indexed row list
                and 0-indexed column list.

Notes:          Add in safeguards to prevent misusage.


Example:
    A = [[0, 1, 2],\
        [ 3, 4, 5],\
        [ 6, 7, 8]];

    B = get_submatrix(A, [0,2], [1,2]) = [[1,2],\ [7,8]]
    
'''

def get_submatrix(mat_in, row_lst, col_lst):
    
    out_submatrix = np.matrix(mat_in[np.ix_(row_lst, col_lst)]);

    return out_submatrix;


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Routines to apply cross-talk corrections
# Written by Juhwan Yoo
'''
Inputs: 
1) ts_qpcr --- (# channels) x (# cycles) e.g. 5 x 75
2) corr_mat --- correlation matrix --- use xtalk_mat_20160504_v1 defined above
3) used_ch_lst --- is the list of channels you wish to use to calculate the
                xtalk deconvolution effect. e.g. if used_ch_list = [0,3,4];
                a submatrix of corr_mat corresponding to those rows and columns
                along with the rows of ts_qpcr of only used_ch_list  are used
                in undoing the effect of cross-talk.
Outputs: 
1) out_mat --- output matrix of same dimensions as ts_qpcr with xtalk effect
            removed via inverse of the cross-correlation matrix.
'''
def apply_xtalk_deconv_v0( ts_qpcr, corr_mat, used_ch_lst ):
    tmp_qpcr = np.matrix(ts_qpcr);
    rowNN = tmp_qpcr.shape[0];
    colNN = tmp_qpcr.shape[1];
    '''
    Note (JY:06/22/16).
    unclear if we should take the inverse of the submatrix, or the
    whole matrix and apply the submatrix.
    '''
    qpcr_submat = get_submatrix(tmp_qpcr, used_ch_lst, range(colNN));
    corr_submat = get_submatrix(corr_mat, used_ch_lst, used_ch_lst);
    corr_submat_inv = np.linalg.inv(corr_submat);

    deconv_qpcr_submat = np.array(corr_submat_inv*qpcr_submat);
    out_mat = np.zeros( (rowNN, colNN) );
    used_ch_ind = 0;

    for row_ii in range(rowNN):
        if row_ii in used_ch_lst:
            out_mat[row_ii,:] = np.array(deconv_qpcr_submat[used_ch_ind,:]);

            used_ch_ind = used_ch_ind + 1;
        else:
            out_mat[row_ii,:] = np.array(tmp_qpcr[row_ii,:]);
    
    return out_mat;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Description:    This section contains functions to assist in 
                normalization.

Authors:		Aditya Rajagopal 

'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def truncFirst(well):
	'''truncate the first sample of the time-series for each channel of the
	   data.'''
	#the truncated time-series is one sample shorter than time-series that
	#generated it
	newWell = np.zeros([np.shape(well)[0], np.shape(well)[1] - 1])

	for i in range(np.shape(newWell)[0]):
		for j in range(np.shape(newWell)[1]):
			newWell[i, j] = well[i, j + 1]

	return newWell

def calcReg(vec, minSample, maxSample):
	'''caculate the regression line for the given vector of data, starting at
	   minSample and proceeding to maxSample. return the slope m, intercept b,
	   and r-value, r.'''
	m, b, r, p, s = st.linregress(np.arange(0, len(vec[minSample:maxSample + 1])), vec[minSample:maxSample + 1])

	return m, b, r

def removeSlope(well, startSample = 4, endSample = 19):
	'''remove the slope for each channel of data.'''
	newWell = np.zeros(np.shape(well))
	#startSample and endSample should at least be 5 samples apart. if they are
	#not, revert to default condition with startSample = 4 and endSample = 19
	if endSample - startSample < 4:
		startSample = 4
		endSample = 19
		
	bestStart = 0
	bestEnd = 0
	bestReg = 0
	for j in np.arange(startSample, endSample - 4):
		for k in np.arange(j + 4, endSample + 1):
			m, b, reg = calcReg(well, j, k)
			if reg > bestReg:
				bestReg = reg
				bestStart = j
				bestEnd = k
	m, b, reg = calcReg(well, bestStart, bestEnd)
	#save the rotated x-value
	newWell = rotateWell(well, np.arctan(m)*180/np.pi)

	return newWell

def rotateCurve(x_old, y_old, theta):
	'''for the given function f(x), x with f(x) = 'y_old' and x = 'x_old', 
	   rotate by the provided angle 'theta'. return the rotated pair:
	   g(x'), x', where (x', g(x')) = M*(x, f(x)) and M is the 2-d rotation
	   matrix. g(x') = 'y_new' and x' = 'x_new'. ''' 
	#x_old and y_old should have the same length
	if np.shape(x_old) != np.shape(y_old):
		print('code_analyze -- rotateCurve: x_old, y_old not same size')
		return -1
	
	#if they do, continue with rotation
	x_new = np.zeros(np.shape(x_old))
	y_new = np.zeros(np.shape(y_old))
	
	for i in range(np.shape(x_old)[0]):
		x_new[i], y_new[i] = rotatePoints(x_old[i], y_old[i], theta)
	
	return x_new, y_new

def rotatePoints(x1, y1, theta):
	'''apply the rotation matrix 'M' for a 2-tuple of (x1, y1). return rotated
	   coordinates (x2, y2), where (x2, y2) = M * (x1, y1).'''
	x2 = np.cos(theta*np.pi/180.0)*x1 + np.sin(theta*np.pi/180.0)*y1
	y2 = -1*np.sin(theta* np.pi/180.0)*x1 + np.cos(theta*np.pi/180.0)*y1
	
	return x2, y2

def rotateWell(well, theta):
	'''rotates curves for the time-series of every channel of the data by the
	   angle 'theta'. the results are returned as an (n x m) vector where 'n' is
	   the number of channels, and 'm' is the number of samples.'''
	newWell = np.zeros(np.shape(well))
	#do this for every sample in the time series (i.e. every 'x' value)
	x_old = np.array(np.arange(0, np.shape(well)[0], 1))
	x_new, newWell = rotateCurve(x_old, well, theta)
	#the shifted x-series is discarded, as such this is not a 'pure'
	#rotation. this is equivalent to subtracting a linear function
	#g(n) from the time series ts(n).
	
	return newWell

def divide_chA_chB(well, chA, chB):
	'''sample-by-sample division of value in chA by the value in chB.'''
	newWell = np.copy(well)
	for i in range(np.shape(well)[0]):
		newWell[chA, i] = well[chA, i]/well[chB, i]
	
	return newWell

def normRox(well):
	'''sample-by-sample division of the value in every channel by the value in
	   the rox channel (channel index 3).'''
	newWell = np.zeros(np.shape(well))
	for i in range(np.shape(well)[0]):
		for j in range(np.shape(well)[1]):
			newWell[i, j] = well[i, j]/well[3, j]

	return newWell

def offset_by_sample(well, n0):
	'''offset the value of every sample in a given channel by the sample n0. do
	   this for every channel.'''
	newWell = np.zeros(np.shape(well))

	#repeat for every channel of data
	for i in range(len(well)):
		newWell[i] = well[i] - well[n0]

	return newWell

def meanFilter(well, wins=[5]):
	newWell = np.copy(well)
	for i in range(len(wins)):
		nextWell = np.zeros(np.shape(well))
		win = wins[i]
		if win%2 == 0:
			win = win + 1

		#repeat for each cycle of data
		for j in range(len(nextWell)):
			#vector to store the window of the samples to be mean-filtered
			winSamples = []
			for k in range(int(win/2) + 1):
				#value at sample itself is stored (center of window)
				if k == 0:
					winSamples.append(newWell[j])
				elif j + k < len(newWell) and j - k >= 0:
					winSamples.append(newWell[j + k])
					winSamples.append(newWell[j - k])
	
			nextWell[j] = np.mean(np.array(winSamples))
		newWell = np.copy(nextWell)
	return newWell

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def diff(well):
	'''compute a simple first derivate for the time-series of each channel of
	   the data.'''
	#first derivative is one sample shorter than the time-series that
	#generated it  
	if len(np.shape(well)) == 2:
		diffWell = np.zeros([np.shape(well)[0], np.shape(well)[1] - 1])
		
		for i in range(np.shape(diffWell)[0]):
			for j in range(np.shape(diffWell)[1]):
				diffWell[i, j] = well[i, j + 1] - well[i, j]
	
		return diffWell
	else:
		diffWell = np.zeros(len(well)-1)
		
		for i in range(len(well)-1):
			diffWell[i] = well[i + 1] - well[i]
	
		return diffWell
		
def boxify(well, cal, threshVal = 0.4):
	'''compute the following pulse-waveform from the first derivative of the 
	   time-series for each channel of data in a well:

	   if time-series[n] > max(cal)*threshVal, then pulse-waveform[n] = 1.
	   else, pulse-waveform[n] = 0'''
	#check to make sure well and cal are the same size
	if not(np.array_equal(np.shape(well), np.shape(cal))):
		print('ccode_analyze.boxify -- well and cal not equal size')
		return -1
	
	#if well and cal are equal in size, continue 
	newWell = np.zeros(np.shape(well))
	for j in range(len(newWell)):
		if well[j] > max(cal[:])*threshVal:
			newWell[j] = 1
		else:
			newWell[j] = 0

	return newWell

def findEdges(well):
	'''returns all the edges for the time-series of each channel of data in a
	   well. this is intended to be used in conjunction with 
	   ccode_analyze.boxify().'''
	edges = []
	#store the parity of the last sample of the time-series
	parity = np.sign(well[len(well) - 1])
	for j in range(len(well)):
		if np.sign(well[len(well) - 1 - j]) != parity:
			edges.append(len(well) - 1 - j)
			parity = np.sign(well[len(well) - 1 - j])

		#corner case is that the first value is one, in which case we want
		#the edge at this cycle.
		elif j == len(well) - 1 and parity == 1:
			edges.append(len(well) - 1)

	# if no edges are found, the last cycle is the edge
	if edges == []:
		edges.append(len(well) - 1)

	return np.array(edges)

def finalEdge(well):
	'''return the ultimate edge of the time-series. this is to be used in
	   conjunction with ccode_analyze.findEdges() and ccode_analyze.boxify().
	'''
	returnEdges = findEdges(well)
	returnFinalEdge = returnEdges[0]
	
	return returnFinalEdge

def index2tuple(index):
	'''converts a well index (ranging from 0 to 95) to a 2-tuple with a row and
	   column for the corresponding location on the 96-well plate. returns the
	   row (ranging from 0 to 7) then the column (ranging from 0 to 11).'''
	return int(index/12), int(index%12)

def tuple2index(row, col):
	'''converts a row (ranging from 0 to 7) and column (ranging from 0 to 11)
	   corresponding to a well in a 96-well plate, to an index (ranging from 0
	   to 95).'''
	return int(12*row + col)

def index2str(index):
	'''converts a index corresponding to a well to a string with the position
	   of that well on a 96-well plate (e.g. a12, e7, b10, etc.).'''
	COL_LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	row, col = index2tuple(index)

	return str(COL_LABELS[col] + str(row + 1))

def tuple2str(row, col):
	'''converts a 2-tuple corresponding to a well to a string with the position
	   of that well on a 96-well plate (e.g. a12, e7, b10, etc.).'''
	return index2str(tuple2index(row, col))

def bin2vec(bin):
	'''given a string that represents a binary number, return this number as 
	   numpy array with each dimension representing a bin of the number. index
	   0 of the array represents the least significant digit.
	   e.g. bin2vec('0111') returns [1, 1, 1, 0]
	        bin2vec('0000') returns [0, 0, 0, 0]
	        bin2vec('1010') returns [0, 1, 0, 1]'''
	returnVec = np.zeros(len(bin))
	for i in range(np.shape(returnVec)[0]):
		returnVec[i] = bin[len(bin) - 1 - i]

	return returnVec
	
def num2bin(num, length = 4):
	ret = bin(num)[2:]
	while len(ret) < length:
		ret = '0' + ret
	return ret

def generateMeans(dataPickle, calib_list, channel_list, algo, dtree):
	'''given a set of calibrators wells and a color channel, generate the
	   intensity means. based on the number of calibrator locations provided,
	   decide how many bins to generate. e.g. if three calibrators provided in 
	   a particular channel, there are two tiers in that channel: NTC, 1i, 2i. 
	   if five are provided, there are four tiers: NTC, 1i, 2i, 4i, 8i. 
	   intermediate levels are either inerpolated or extrapolated as the sums
	   of the specified calibrator levels.'''
	ret = []
	
	zeroVals = np.zeros(len(calib_list))
	calValues = []
	e15 = []
	for ind in range(len(calib_list)):
		calibs = calib_list[ind]
		channel = channel_list[ind]
		#force calib to be an np array (as opposed to a list)
		calibs = np.array(calibs)
		calValues.append(np.zeros(len(calibs)))
		e15.append(np.zeros(len(calibs)))
		#generate values for synthesizing means
		zeroVals[ind] = analyzeWell([dataPickle['well_list'][int(calibs[0])]['data_array'], [channel], [[]], algo, dtree])[0][1][0]
		
	for ind in range(len(calib_list)):
		calibs = calib_list[ind]
		channel = channel_list[ind]
		for j in range(len(calibs)):
			res = analyzeWell([dataPickle['well_list'][int(calibs[j])]['data_array'], [channel], [[]], algo, dtree])
			calValues[ind][j] += res[0][1][0]
			e15[ind][j] = res[0][2]
		intensityMeans = np.zeros(2**(len(calibs)-1))
		for i in range(len(intensityMeans)):
			#generate masks for every intensity level
			strBinMask = np.binary_repr(np.left_shift(i, 1), 
				                        width = np.shape(calibs)[0])
			intensityMeans[i] = np.dot(bin2vec(strBinMask), calValues[ind])
			numbin = num2bin(i)
			for j in range(len(numbin)):
				if len(e15[ind]) < 5 and j == 0:
					continue
				if numbin[j] == '1':
					intensityMeans[i] += e15[ind][-1-j+5-len(e15[ind])]
			if not i in [0, 1, 2, 4, 8] and len(calib_list[ind]) > 4:
				intensityMeans[i] -= min(np.mean(e15[ind]), intensityMeans[i] - intensityMeans[i-1] - intensityMeans[1]/2)
		ret.append(intensityMeans)
	
	return ret

def generateRealMeans(dataPickle, calib_list, channel_list, algo = False, dtree = True):
	'''aeafdafa'''
	ret = []
	for ind in range(len(calib_list)):
		calibs = calib_list[ind]
		channel = channel_list[ind]
		means = np.zeros(len(calibs))
		for j in range(len(calibs)):
			res = analyzeWell([dataPickle['well_list'][int(calibs[j])]['data_array'], [channel], [[]], algo, dtree])
			means[j] = res[0][1][0]
		ret.append(means)

	return ret

def generateFracBins(means, frac = 1.0/3):
	'''generate acceptance bins with the '1/3, 1/3, 1/3' spacing criteria.'''
	returnBins = np.zeros([np.shape(means)[0], 2])
	for i in range(np.shape(returnBins)[0]):
		#make upper bin
		#exception: if we are at the highest intensity level, the upper acceptance
		#bin is the same size as the lower acceptance bin 
		if i == np.shape(returnBins)[0] - 1:
			returnBins[i, 1] = means[i] + (means[i] - means[i - 1])*frac
		else:
			returnBins[i, 1] = means[i] + (means[i + 1] - means[i])*frac
		#make lower bin
		#exception: if we are at the lowest intensity level, the lower acceptance
		#bin is the same size as the upper acceptance bin
		if i == 0:
			returnBins[i, 0] = means[i] - (means[i + 1] - means[i])*frac
		else:
			returnBins[i, 0] = means[i] - (means[i] - means[i - 1])*frac

	returnBins[0, 1] = returnBins[1, 0] - 0.01
	returnBins[0, 0] = -1
	return returnBins

def generateNWSBins(means):
	'''generate acceptance bins with no white space between levels. for a given
	   intensity level 'n', the upper bin is the midpoint between the mean for
	   level 'n' and level 'n+1'. the lower bin is the midpoint between the 
	   mean at level 'n' and level 'n-1'. for the lowest intensity tier, the
	   lower bound is a very large negative number. for the highest intensity
	   tier the upper bound is very large positive number.'''
	returnBins = np.zeros([np.shape(means)[0], 2])
	#an unreasonably large number
	CONS_MUL = 10000.0
	for i in range(np.shape(returnBins)[0]):
		#make upper bin
		#exception: if we are at the highest intensity level, the upper acceptance
		#bin should be an unreasonably high number (practically +Inf) 
		if i == np.shape(returnBins)[0] - 1:
			returnBins[i, 1] = CONS_MUL*np.max(means)
		else:
			returnBins[i, 1] = means[i] + (means[i + 1] - means[i])/2.0
		#make lower bin
		#exception: if we are at the lowest intensity level, the lower acceptance
		#bin should be an unreasonably low number (practically -Inf)
		if i == 0:
			returnBins[i, 0] = -1*CONS_MUL*np.max(means)
		else:
			returnBins[i, 0] = means[i] - (means[i] - means[i - 1])/2.0
	returnBins[0][1] = means[1]*2/3
	returnBins[1][0] = means[1]*2/3

	return returnBins

def checkOverlaps(bins):
	'''given a set of bins, check to see if any of the levels overlap. the bins
	   vector should be dimensioned (m x n x 2), where 'n' is the number of
	   distinct intensity levels, and 'm' is the number of color channels. a 
	   vector the size of bins is returned with the value of 1, if there is an
	   overlap for that element, or a 0 if there is not. by contruction, 
	   bins[m, 0, 0] and bins[m. n - 1, 1] will always be zero.'''
	newBins = np.zeros(np.shape(bins))
	for i in range(np.shape(bins)[0]):
		for j in range(1, np.shape(bins)[1] - 1):
			if bins[i, j, 1] > bins[i, j + 1, 0]:
				newBins[i, j, 1] = 1
				newBins[i, j + 1, 0] = 1
			if bins[i, j, 0] < bins[i, j - 1, 1]:
				newBins[i, j, 0] = 1
				newBins[i, j - 1, 1] = 1
	
	return newBins
	
def findBump(twoder):
	mins = argrelextrema(twoder[20:70], np.less, order = 5)[0]
	maxes = argrelextrema(twoder[10:65], np.greater, order = 5)[0]
	
	max2der = max(abs(twoder))
	realmins = []
	realmaxes = []
	thresh = 0.00
	for i in range(len(mins)):
		ind = mins[i] + 20
		if twoder[min(len(twoder)-1,ind+6)] - twoder[ind] > thresh*max2der and twoder[ind-6] - twoder[ind] > thresh*max2der:
			realmins.append(ind)
	if twoder[-1] < 0 and twoder[-3] < 0:
		realmins.append(len(twoder)-1)
	for i in range(len(maxes)):
		ind = maxes[i] + 10
		if twoder[ind] - twoder[min(len(twoder)-1,ind+6)] > thresh*max2der and twoder[ind] - twoder[ind-6] > thresh*max2der:
			realmaxes.append(ind)
	
	if len(realmaxes) <= 1:
		return -1
	startLoc = -1
	for i in range(len(realmins)):
		if realmins[-i-1] < realmaxes[1]:
			startLoc = (realmins[-i-1] + realmaxes[1])/2
			break
	midLoc = -1
	for i in range(len(realmins)):
		if realmins[i] > realmaxes[1]:
			midLoc = (realmins[i] + realmaxes[1])/2
			break
	return [startLoc, midLoc]

def dTree(well, means):
	twoder = meanFilter(diff(diff(well)),[5,5])
	bumps = findBump(twoder)
	avg = (well[-1]-well[0])/75
	posBins = range(16)
	if len(means) > 1 and bumps != -1 and well[-1] >= means[1]:
		if well[-1] < means[3] and max(twoder[bumps[0]:len(twoder)]) < 0:
			return [well[-1] - well[15], posBins, True]
		bumpVal = well[bumps[0]]
		bumpI, bumpE = 0, 0
		minDist = 5000
		for e in range(4):
			if 2**e >= len(means):
				break
			dist = abs(bumpVal - means[2**e])
			if dist < minDist:
				bumpE = e
				bumpI = 2**e
				minDist = dist
		posBins = []
		for e in range(len(means)):
			if e >= bumpI and num2bin(e)[-1-bumpE] == '1':
				posBins.append(e)
		if bumpI == 0:
			posBins = [1,2]+posBins[1:]
				
		predVal = well[-1]
		
		if diff(well)[-1] > avg/3:
			if bumps[1] == -1:
				predVal = max(predVal,bumpVal + 2*(predVal - bumpVal))
			else:
				turnLoc = bumps[1]
				predVal = max(predVal,bumpVal + 2*(well[turnLoc] - bumpVal))
		return [predVal, posBins, predVal == well[-1]]
	return [well[-1], posBins, False]
	
def matchBin(value, posBins, flag, means):
	retBin= posBins[0]
	minDist = 2*abs(value - means[posBins[0]])
	for i in range(1, len(posBins), 1):
		pen = 1
		if flag and means[posBins[i]] < value:
			pen = 2
		dist = pen*abs(value - means[posBins[i]])
		if dist < minDist:
			retBin = posBins[i]
			minDist = dist
	return retBin
	
def analyzeWell(inputs):
	'''analyze the well and return a normalized time-series, and the
	   interpreted fluorescence cycle and value at that cycle.'''
	#first, read in data
#	well = dataPickle['well_list'][index]['data_array']
	#truncate the first cycle
	well, channels, means, algo, dtree = inputs
	wellT = truncFirst(well)
	corWell = wellT
	if len(channels) > 1:
		corWell = apply_xtalk_deconv_v0(wellT, xtalk_mat_20160504_v1, channels)
	
	ret = []
	for i in range(len(channels)):
		if algo or not dtree:
			ret.append(analyzeChannel(normRox(corWell)[channels[i]], -1, algo, dtree))
		else:
			ret.append(analyzeChannel(normRox(corWell)[channels[i]], means[i], algo, dtree))
	return ret
	#normalize by ROX and remove the DC offset in fluorescence, then smooth
	#using a mean filter with five samples
	
def analyzeChannel(well, means, algo, dtree):
	filteredWell = offset_by_sample(meanFilter(well,[9,9,9,9]),0)
	newWell = filteredWell
	newWell = removeSlope(newWell)
	e15 = newWell[10]
	newWell = offset_by_sample(newWell, 10)
	if algo:
		diffNewWellBox = boxify(diff(newWell), diff(newWell), 0.15)
		edge = finalEdge(diffNewWellBox)
		return [newWell, [newWell[edge+1], range(16), False], e15, edge]
	if dtree:
		return [newWell, dTree(newWell, means), e15, len(newWell)-1]
	return [newWell, [newWell[-1], range(16), False], e15, len(newWell)-1]
	
def optimizePlate(values, means):
	bestResult = 200
	retVals = np.zeros(len(values))
	for i in range(len(values)):
		if values[i] != -1:
			retVals[i] = values[i][0]
	retMeans = means
	#if type(retMeans) is int:
	return [retVals, retMeans]
#	cap = 20
	offset = 0
	if len(means) > 8:
#		cap = 400
		m, b, r, p, s = st.linregress([1,2,4], [means[1],means[2],means[4]])
		offset = means[8] - (b + 8*m)
	zeroVal = 0
		
	lowShift, highShift = 0, 0
	newMeans = np.copy(means)
	for i in [3, 5, 6]:
		newMeans[i] -= lowShift*zeroVal
	newMeans[7] -= (1+lowShift)*zeroVal
	
	if len(means) > 8:
		for i in range(9, 16, 1):
			newMeans[i] -= min(offset, newMeans[i]-newMeans[i-1]-newMeans[1]/2)
	binsFrac = generateFracBins(newMeans, frac=1.0/9)
#	if isOverlap(binsFrac):
#		continue
#	for i in range(1, len(newMeans)-1, 1):
#		rat = (newMeans[i+1]-newMeans[i])/(newMeans[i]-newMeans[i-1])
#		if rat > 3 or rat < 1.0/3:
#			continue
	fails = 0
	for i in range(len(values)):
		if values[i] == -1:
			continue
		if values[i][1][0] == 0 and binResult(values[i][0], binsFrac) == -1:
			fails += 1
			
	if fails < bestResult:
		for i in range(len(values)):
			if values[i] == -1:
				retVals[i] = -1
				continue
			if values[i][1][0] == 0:
				retVals[i] = values[i][0]
			else:
				retVals[i] = newMeans[matchBin(values[i][0], values[i][1], values[i][2], newMeans)]
		retMeans = newMeans
		bestResult = fails
	return [retVals, retMeans]

###PUBLIC FUNCTIONS --- THESE ARE THE ONLY FUNCTIONS YOU SHOULD CALL
def analyzePlate(pickle, calibs, channels, algo = False, dtree = True, numChannels = 5):
	dataPickle = importData(pickle)
	
	if (algo or (not dtree)) and len(channels) < numChannels:
		channels = range(numChannels)
	
	#generate means for fam channel
	means = []
	analResults = []
	curves = []
	edges = []
	opt = []
	if algo or not dtree:
		means = -1
	else:
#		means = generateMeans(dataPickle, calibs, channels, algo, dtree)
		means = generateRealMeans(dataPickle, calibs, channels, algo, dtree)
	for i in range(len(channels)):
		analResults.append([])
		curves.append([])
		opt.append([])
		edges.append([])
	
	NUM_WELLS = 96
	inputs = []
	results = []
	#do analysis for all 96 wells
	for i in range(NUM_WELLS):
#		print('ANALYZING WELL '+str(i))
		inputs.append([dataPickle['well_list'][i]['data_array'], channels, means, algo, dtree])
	pool = Pool()
	results = pool.map(analyzeWell, inputs)
	pool.close()
	pool.join()
	
	for i in range(NUM_WELLS):
		for j in range(len(results[i])):
			analResults[j].append(results[i][j][1])
			curves[j].append(results[i][j][0])
			edges[j].append(results[i][j][3])
#	print('OPTIMIZING')
	for j in range(len(channels)):
		if algo or not dtree:
			opt[j] = optimizePlate(analResults[j], -1)
		else:
			opt[j] = optimizePlate(analResults[j], means[j])
	
	return [curves, opt, edges]
	
def importData(file):
	'''imports a plate of data from the pickle structure used to store it, into a
	   dictionary format that can be easily accessed and interrogated. if the
	   pickle to import is not found, prints an error message, and generates an
	   exception.'''
	try:
		return pickle.load(open(str(file), 'rb'))
	except:
		print('ccode_analyze.importData -- pickle file not found')
		return -1

def isOverlap(bins):
	'''if there is an overlap between levels in the binning vector that is
	   passed, return "true". else, return "false". bins is a (n x 2) vector,
	   where 'n' is the number of disinct intensity levels.'''
	for i in range(len(bins)-1):
		if bins[i][1] > bins[i+1][0]:
			return True
	return False

def binResult(val, bins):
	'''for a passed value, 'val', report which intensity bin it falls into.
	   the bins are provided as an (nx2) vector. if 'val' falls outside of any
	   provided bins, -1 is returned. else, 'n' is returned.'''
	for i in range(np.shape(bins)[0]):
		#in case bins are adjacent, we accept the lower bin edge and exclude
		#the upper bin edge
		if val >= bins[i, 0] and val < bins[i, 1]:
			return i

	#in case 'val' does not fall into any bin, return -1
	return -1

def plotWell(time_series, edges, channels, calls, outputFig, outputFile):
	'''plotting routine'''
	MAX_CHANS = 6
	FIG_COLS = 2
	FIG_ROWS = 3
#	INCHES_PER_SUBPLOT = 10
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

#	fig1 = plt.figure(figsize=(FIG_ROWS*INCHES_PER_SUBPLOT, FIG_COLS*INCHES_PER_SUBPLOT))
	style.use('ggplot')
	outputFig.clf()

	for i in range(len(channels)):
		ax = outputFig.add_subplot(FIG_ROWS, FIG_COLS, i + 1)
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

		if edges[i] != -1:
			ax.plot(time_series[i], color = calls[i], linewidth = 2.0)
			ax.plot(edges[i], time_series[i][edges[i]], marker = marker_shape, linewidth = 3.0, color = calls[i])
			ax.set_title(CHANNEL_TAG + str(channels[i] + 1), fontsize = 10.0)
			ax.set_xlabel('cycles', color = [0.4, 0.4, 0.4], fontsize = 12.0)
			ax.set_ylabel('intensity', color = COLORS[channels[i]], fontsize = 12.0)
			for tl in ax.get_yticklabels():
				tl.set_color(COLORS[channels[i]])

			for tl in ax.get_xticklabels():
				tl.set_color(COLORS[channels[i]])

	outputFig.savefig(outputFile, format = 'png', dpi = 150)
	outputFig.clf()
	plt.cla()

#class WellThread(threading.Thread):
#	def __init__(self, well, channels, means, algo, dtree, results, index):
#		threading.Thread.__init__(self)
#		self.well = well
#		self.channels = channels
#		self.means = means
#		self.algo = algo
#		self.dtree = dtree
#		self.results = results
#		self.index = index
#		
#	def run(self):
#		res = analyzeWell(self.well, self.channels, self.means, self.algo, self.dtree)
#		self.results[self.index] = res
#		print('ANALYZED WELL '+str(self.index))
