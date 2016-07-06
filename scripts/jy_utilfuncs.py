#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
author:             Juhwan Yoo
description:        This file contains a preliminary function library for use 
                    with chromacode import methods.
revision
history:
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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

import kparam_funcs_v2 as kpf2

import copy
import os
import pickle
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# return list of files in a directory
def get_dirfiles( dir_name ):
    dir_flist = os.listdir( dir_name );
    return dir_flist;
'''
filt_str is the portion of the string that needs to match to create
the total filtered list.
'''
def get_matching_dirfile( dir_name, filt_str ):
    dir_flist = get_dirfiles( dir_name );
    tmp_flist = [];

    for fname_ii in dir_flist:
	if (filt_str in fname_ii):
	    tmp_flist.append(fname_ii);
	
    return tmp_flist;


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
format conversion functions
desciprtion: converts a pklfile with file name pklname to a matlab matfile.

notes: implement errorchecking for pklname to be both a string and also 
        having extension '.pkl'.

        - this function seems to have trouble converting lists of objects, i.e. lists of dictionaries.

'''

def convert_pkl2mat( pklname, specify_outfile=0, matname = 'matsavename.mat' ):
    tmp_pklfile = open( pklname, 'rb');
    tmp_data = pickle.load(tmp_pklfile);
    tmp_pklfile.close();
    if specify_outfile:
        # user requested to specify output filename so use matname
        #scipy.io.savemat(matname, tmp_data);
        fname_tmp = matname;
    else:
        #print 'no output';
        # automatically generate a name.
        fname_tmp=replace_fextension( pklname, '.mat');
    
    scipy.io.savemat(fname_tmp, tmp_data);
    print 'completed converting file: %s to %s' % (pklname, fname_tmp);
    return fname_tmp;

'''
description: retrieves the file extension in a string (defined by all the characters after the last '.' in the string and inclusive of the last '.'.
'''
def get_fextension(fname):
    if isinstance(fname, str):
        tmp_ext_ind = get_ext_loc(fname);
        if tmp_ext_ind == -1:
            print 'no extension found'
            tmp_ext = '';
        else:
            tmp_ext = fname[tmp_ext_ind:];
    else:
        print 'input fname = '; print fname
        print 'has no extension';
        tmp_ext = ''
    return tmp_ext;



    return tmp_fext;
'''
description: function replaces input string fname comprising string fname = fname_main + ',' + f_extension tp fname = fname_main + '.' o_ext where '.' is the terminal '.' in the string fname;
'''
def replace_fextension(fname, o_ext):
    if isinstance(fname, str):
        tmp_ext_ind = get_ext_loc(fname);
        if tmp_ext_ind == -1:
            if o_ext[0] is '.':
                fname_out = fname + o_ext;
            else:
                fname_out = fname + '.' + o_ext;
        else:
            fname_main = fname[0:tmp_ext_ind];
            if o_ext[0] is '.':
                fname_out = fname_main + o_ext;
            else:
                fname_out = fname_main + '.' + o_ext;
    else:
        print 'fname = '; print fname; print 'is not a string';
        fname_out = '';
        # input was not a string, 
    return fname_out;

'''
description: obtains location of where the file extension starts.
            if there is no '.' in the string or if the input is not
            a string, the output of the function is -1;
'''
def get_ext_loc(fname):
    if isinstance(fname, str):
        tmp_pt_arr = np.where(np.array(list(fname)) == '.')[0];
        if (len(tmp_pt_arr) > 0):
            tmp_loc = tmp_pt_arr[-1];
        else:
            # output inf to indicate that no '.' were found.
            tmp_loc = -1;
            

    else:
        print fname
        print 'is not a string'
        tmp_loc = -1;
    return tmp_loc;


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
generic functions for file handling etc....
'''
def ensure_dir(f):
    d = os.path.dirname(f);
    if not os.path.exists(d):
        os.makedirs(d);
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def sel_list_elems(lst, lst_inds):
    return [lst[ii] for ii in lst_inds]; 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def set_list_elems(lst, lst_inds, lst_vals):
    # add assertion that enforces that length(list_inds) = length(lst_vals)
    lst_cp = copy.deepcopy(lst);
    if (np.size(lst_inds) == np.size(lst_vals)):
        for sel_ii in range(np.size(lst_inds)):
            ch_ind = lst_inds[sel_ii];
            lst_cp[ch_ind] = lst_vals[sel_ii];
        return lst_cp;
    else: 
        return None;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_list_elems(lst, lst_inds):
    tmp_list_elems = [lst[ii] for ii in lst_inds];
    return tmp_list_elems;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def find_inds_inrange(lst, max_bound, min_bound):
    #-note, max_vals must be greater than min_vals.
    #-
    below_inds = set(np.where( np.array(lst) < max_bound )[0]);
    above_inds = set(np.where( np.array(lst) > min_bound )[0]);
    sat_inds = list( (below_inds & above_inds) );

    return sat_inds;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

