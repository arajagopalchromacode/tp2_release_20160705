import numpy as np
import scipy as sp
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

import csv
import xlrd
import datetime
import mimetypes
import os
import pickle

import ccode_lib_import_utilfuncs as cif

'''
author:             Juhwan Yoo
description:        temporary file to facilitate analysis of data collected 
                    on 2/22/16  
'''

def get_viia7_plateData(xlsheet_rawdata, dSheet_Info={'cyc_col':2, 'ch1_col':3, \
    'ch2_col':4, 'ch3_col':5, 'ch4_col':6, 'ch5_col':7}, rowsNN=8, colsNN=12):
    plate_data = list();
    for well_ii in range(rowsNN*colsNN):
        tmp_well_data = cif.viia7_get_WellData(xlsheet_rawdata, well_ii, dSheet_Info, rowsNN, colsNN);
        plate_data.insert(well_ii, tmp_well_data);

    return plate_data;

'''
description: obtains the location of the following data within each excel file.

    Data Obtained: 
    1) plate_id
    2) exp_end_time
    3) well_list
        1) well_dictionary x 96
            1) well_index
            2) well_tag
            3) cyc
            4) ch1
            5) ch2
            6) ch3


    1) sheet name, i.e. raw data, sample setup, etc...
    2) data label (string like column headers) (row/column) index, both 0-indexed
    3) starting data location (relative to column) (row/column index)
    4) data type
    
    dependencies:
    1) assert value of the obtained values from the sheet, otherwise flag an error.
    For example, if one retrieves a time series it should be numerical and non-empty.
'''

def get_viia7_plateDataMeta0(xlsf, dSheet_Info={'cyc_col':2, 'ch1_col':3, \
    'ch2_col':4, 'ch3_col':5, 'ch4_col':6, 'ch5_col':7}, rowsNN=8, colsNN=12):

    '''
    keep this place open for plateID
    '''
    #plateID_str = cif.xls_get_PlateID(xlsheet_rawdata);
    #user_str = cif.xls_get_username(xlsheet_rawdata);
    chem_str = cif.xls_get_Chemistry(xlsheet_rawdata);
    filename_str = cif.xls_get_FileName(xlsheet_rawdata);
    blocktype_str = cif.xls_get_BlockType(xlsheet_rawdata);
    exp_end_time = cif.xls_get_ExperimentStartTime(xlsheet_rawdata);
    #exp_start_time=cif.xls_get_ExperimentEndTime(xlsheet_rawdata);
    # barcode_str=cif.xls_get_Barcode(xlsheet_sampleSetup);


    return None;
'''
xlsf is the xlsheet object

i.e. 
xlsf = xlrd.open_workbook(spreadsheet_fname);
sh_rawdata = xlsf.sheet_by_name('Raw Data');
sh_sampsetup = xlsf.sheet_by_name('Raw Data');

notes:
    -this procedure needs to be revised to accelerate file conversion

'''
def get_viia7_plateDataAll(xlsf, sname='tmp_data', parse_Info={'cyc_col':2}):
    sh_rawdata = xlsf.sheet_by_name('Raw Data');
    sh_sampsetup = xlsf.sheet_by_name('Sample Setup');
    '''
    The following information is not currently collected:
    1) exp_user
    2) plate_id
    3) 
    '''
    plate_dict={'exp_fname':'No Entry', \
        'exp_user':'No Entry', \
        'exp_date':'No Entry', \
        'exp_st_time':'No Entry', \
        'exp_end_time':'No Entry', \
        'instr_type':'No Entry', \
        'instr_name':'No Entry', \
        'instr_SN':'No Entry', \
        'block_type':'No Entry', \
        'chemistry':'No Entry', \
        'passive_ref':'No Entry', \
        'plate_id':'No Entry', \
        'well_list':[]};
    fname_str =         cif.xls_get_FileName( sh_rawdata );
    exp_date_str =      'NA';
    exp_st_str =        cif.xls_get_ExperimentStartTime( sh_rawdata );
    exp_end_str =       cif.xls_get_ExperimentEndTime( sh_rawdata );
    instr_type_str =    cif.xls_get_InstrType( sh_rawdata );
    instr_name_str =    cif.xls_get_InstrName( sh_rawdata );
    instr_SN_str =      cif.xls_get_InstrSN( sh_rawdata );
    block_type_str =    cif.xls_get_BlockType( sh_rawdata );
    chem_str =          cif.xls_get_Chemistry( sh_rawdata );
    pass_ref_str =      cif.xls_get_PassRef( sh_rawdata );
    # call the other function to get well_info.
    # load meta data
    plate_dict['exp_fname'] = fname_str;
    plate_dict['exp_st_time']=exp_st_str;
    plate_dict['exp_end_time']=exp_end_str;
    plate_dict['instr_type']=instr_type_str;
    plate_dict['instr_name']=instr_name_str;
    plate_dict['instr_SN']=instr_SN_str;
    plate_dict['block_type']=block_type_str;
    plate_dict['chemistry']=chem_str;
    plate_dict['passive_ref']=pass_ref_str;

    # now extract all well info
    tmp_well_list = get_viia7_plateData(sh_rawdata);
    plate_dict['well_list']=tmp_well_list;

    return plate_dict;

def convfile_viia7( xlfile_name, opts_info={'data_dir':'', 'save_dir':'', \
        'save_name':'tmp_save_name'}):
    if (opts_info['data_dir'] == ''):
        opts_info['data_dir'] == os.getcwd() + '/';
    
    if (opts_info['save_dir'] == ''):
        opts_info['save_dir'] == os.getcwd() + '/';

    tmp_xlfile_name = opts_info['data_dir']+xlfile_name;
    
    xlsheet_viia7 = xlrd.open_workbook(tmp_xlfile_name);
    tmp_savfile_name = opts_info['save_dir']+opts_info['save_name'];

    plate_dict = get_viia7_plateDataAll(xlsheet_viia7);
    afile = open( tmp_savfile_name, 'wb');
    pickle.dump(plate_dict, afile);
    afile.close();
    
    
    return tmp_savfile_name;


def convfile_viia7_2( xlfile_name, opts_info={'data_dir':'', 'save_dir':'', \
        'save_name':'tmp_save_name', 'save2pkl':True, 'ret_xlsworkbook':False}):
    #if (opts_info['data_dir'] == ''):
    #    opts_info['data_dir'] == os.getcwd() + '/';
    
    #if (opts_info['save_dir'] == ''):
    #    opts_info['save_dir'] == os.getcwd() + '/';
        #   opts
    #if (opts_info['save_name']==''):
    #    opts_info['save_name'] = xlfile_name;
   
    tmp_xlfile_name = opts_info['data_dir']+xlfile_name;
    print tmp_xlfile_name 
    xlsheet_viia7 = xlrd.open_workbook(tmp_xlfile_name);
    tmp_savfile_name = opts_info['save_dir']+opts_info['save_name'] + '.pkl';
    print opts_info['save_name'];
    plate_dict = get_viia7_plateDataAll(xlsheet_viia7);

    out_results = {'data':plate_dict} #, 'save_fname':''};
    print tmp_savfile_name;
    if opts_info['save2pkl']==True:
        afile = open( tmp_savfile_name, 'wb');
        pickle.dump(plate_dict, afile);
        afile.close();
        out_results['save_fname']=tmp_savfile_name;
    
    #if opts_info['ret_xlsworkbook']==True:
    #    out_results['xls_workbook'] = xlsheet_viia7;

    return out_results
    #return tmp_savfile_name;


#def get_viia7_plateDataMeta(xlsfile, dSheet_Info{'cyc_col':2, 'ch1_col':3



#def get_viia7_plateDataAll(xlshee_rawdata, dSheet_Info={'cyc_col':2, 'ch1_col':3, \
#    'ch2_col':4, 'ch3_col':5, 'ch4_col':6, 'ch5_col':7}, rowsNN=8, colsNN=12):
    
#    return None;



def get_sds7500_plateData(xlsheet_rawdata, dSheet_Info={'cyc_col':2, \
    'ch1_col':3, 'ch2_col':4, 'ch3_col':5, 'ch4_col':6, 'ch5_col':7}, \
    rowsNN=8, colsNN=12):


    return None;
