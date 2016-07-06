'''
Description:    This file contains functions for signal processing
                operations required for qpcr curve analysis,
                parameter extraction, and quantitation.

            

'''
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import xlrd
import ccode_norm as cn
import ccode_import as ci
import string
import datetime
import mimetypes
import scipy as sp
from scipy import signal
import scipy.io

import copy
import os
import pickle

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Description:    This section contains functions to assist in 
                normalization and cross-talk cancellation.

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
# Routines to apply corrections
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
    deconv_rNN = deconv_qpcr_submat.shape;
    out_mat = np.zeros( (rowNN, colNN) );
    used_ch_ind = 0;

    for row_ii in range(rowNN):
        if row_ii in used_ch_lst:
            used_ch_jj = used_ch_lst[used_ch_ind];
            out_mat[row_ii,:] = np.array(deconv_qpcr_submat[used_ch_ind,:]);

            used_ch_ind = used_ch_ind + 1;
        else:
            out_mat[row_ii,:] = np.array(tmp_qpcr[row_ii,:]);
    
    return out_mat;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Inputs:
1) chmat_in     ---input matrix assumed to have len(rows)=# channels
                and len(columns) to be equal to the length of the 
                time series.

2) rox_ind      ---the index of the channel (0-indexed, i.e. 0 = ch1
                ---to normalize element-by-element all rows by.
'''

def normalize_matbyrox( chmat_in, rox_ind ):
    # chmat_in is the in put matrix of 5 channels
    # rox_ind gives the channel for which the normalization is to occur
    ch_rox = np.array(chmat_in[rox_ind,:]);
    tmp_ch1 = np.array(chmat_in[0,:]);
    tmp_ch2 = np.array(chmat_in[1,:]);
    tmp_ch3 = np.array(chmat_in[2,:]);
    tmp_ch4 = np.array(chmat_in[3,:]);
    tmp_ch5 = np.array(chmat_in[4,:]);

    out_ch1 = np.divide(tmp_ch1, ch_rox);
    out_ch2 = np.divide(tmp_ch2, ch_rox);
    out_ch3 = np.divide(tmp_ch3, ch_rox);
    out_ch4 = np.divide(tmp_ch4, ch_rox);
    out_ch5 = np.divide(tmp_ch5, ch_rox);

    out_mat = np.matrix([out_ch1, out_ch2, out_ch3, out_ch4, out_ch5]);
    
    return out_mat;