'''
filename:       ccode_lib_import.py
description:    top-level import functions.

'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
# basic import dependent functions
import os
import datetime
import pickle
import string
import xlrd
import mimetypes
import csv

import analysis_utilfuncs as aif

# Unnecessary but occasionaly useful definitions

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

def import_raw_data2pkl( fname, opts={'verbose':True, 'save2pkl':True, \
        'data_dir':'', 'save_dir':'', 'save_name':''} ):
    tmp_opts_info = {};
    tmp_opts_info['data_dir'] = opts['data_dir'];
    tmp_opts_info['save_dir'] = opts['save_dir'];
    tmp_opts_info['save_name'] = opts['save_name'];
    tmp_opts_info['save2pkl'] = opts['save2pkl'];
    tmp_opts_info['ret_xlsworkbook'] = False;


    # for now, this wrapper only supports viia7 machines.
    converted_data = aif.convfile_viia7_2( fname, tmp_opts_info );
    
    
    return converted_data;



