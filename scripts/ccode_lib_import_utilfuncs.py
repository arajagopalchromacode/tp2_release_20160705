'''
author: Juhwan Yoo

'''

import string;
import math;
import numpy as np;
import pickle;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# creating functions to mine 


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# functions to read columns and rows in from excel sheet
'''
'''
# this is already provided
'''
def get_xls_rowN( data_sheet );
    return None;
'''
'''
'''
# this is already provided within xlrd
'''
description:
'''
'''
def get_xls_colN( data_sheet );
    return None;
'''

'''
just a brief aside, how to check default arguments.
DEFAULT = object()
def foo(param=DEFAULT):
    if param is DEFAULT:
        ...
Usually you can just use None as the default value, if it doesn't make sense as a value the user would want to pass.

The alternative is to use kwargs:

def foo(**kwargs):
    if 'param' in kwargs:
        param = kwargs['param']
    else:
        ...
'''


def search_allxlscells( xlsfile, search_opts={'noargs':True} ):


    return None;


def write_importinfo():
    return None;

def read_importinfo():
    return None;


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_xls_colByNum( data_sheet, col_ind, row_ind=0):
    rowN = data_sheet.nrows;
    row_inds = range(row_ind,rowN);
    #colN = data_sheet.ncols;
    # output a list, because we don't know if cell data 
    # is numerical so can't use array.
    tmp_arr_vals= list();
    tmp_arr_ctypes = list();
    for row_ind in row_inds:
	tmp_val = data_sheet.cell(row_ind, col_ind).value;
	tmp_ctype= data_sheet.cell(row_ind, col_ind).ctype;
	STR_CTYPE = 1;
	NUM_CTYPE = 2;
	if tmp_ctype == NUM_CTYPE:
	    tmp_arr_vals.append( tmp_val );
            tmp_arr_ctypes.append( tmp_ctype );
	elif tmp_ctype == STR_CTYPE:
            tmp_arr_vals.append(str(tmp_val));
            tmp_arr_ctypes.append(tmp_ctype);
        elif len(tmp_val) == 0:
            tmp_arr_vals.append( tmp_val );
            tmp_arr_ctypes.append( tmp_ctype );
        else:
            tmp_arr_vals.append(tmp_val);
	    tmp_arr_ctypes.append(tmp_ctype);
        # end if
    # end for
    col_out = {"vals":tmp_arr_vals, "ctypes":tmp_arr_ctypes};
    return col_out

def set_xls_colByNum( data_sheet, col_ind, row_ind=0 ):
    return None;


'''
'''
def get_xls_colByName( data_sheet,):
    return None;

'''
description: find_firstDataRow finds the first non-empty cell in a column 
(col_num) within a given (data_sheet) after/inclusive of the row denoted by 
row_offset. Note, row/columns are 0 indexed.
'''
def find_firstDataRow( data_sheet, col_num, row_offset):
    return None;
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function to convert raw data time formats to a universal standard. Should be a numerical value.

def search_cells( data_sheet, search_str ):
    # it is assumed data sheet is xlsheet.sheet_by_name('sheet_name')
    rowNN = data_sheet.nrows;
    colNN = data_sheet.ncols;
    # -hit_inds contains row index col index pairs for locations corresponding
    # to locations where the search term was relevant.
    hit_inds = list();

    for row_ii in range(rowNN):
        for col_ii in range(colNN):
            # tmp_cell_type = 1 if string and 2 if number
            tmp_cell_type = data_sheet.cell(row_ii,col_ii).ctype;
            tmp_cell_val = data_sheet.cell(row_ii,col_ii).value;
            #print tmp_cell_type
            #print tmp_cell_val
            if tmp_cell_val == search_str:
                hit_inds.append([row_ii,col_ii]);
            
    return hit_inds;

def search_cells_strip( data_sheet, search_str ):
    # it is assumed data sheet is xlsheet.sheet_by_name('sheet_name')
    rowNN = data_sheet.nrows;
    colNN = data_sheet.ncols;
    # -hit_inds contains row index col index pairs for locations corresponding
    # to locations where the search term was relevant.
    hit_inds = list();

    for row_ii in range(rowNN):
        for col_ii in range(colNN):
            # tmp_cell_type = 1 if string and 2 if number
            tmp_cell_type = data_sheet.cell(row_ii,col_ii).ctype;
            tmp_cell_val = str(data_sheet.cell(row_ii,col_ii).value).strip();
            #tmp_cell_val = str(tmp_cell_val);
            tmp_cell_val.strip();
            #print tmp_cell_type
            #print tmp_cell_val
            if tmp_cell_val == search_str:
                hit_inds.append([row_ii,col_ii]);
            
    return hit_inds;

def search_cells_nocase( data_sheet, search_str ):
    # it is assumed data sheet is xlsheet.sheet_by_name('sheet_name')
    rowNN = data_sheet.nrows;
    colNN = data_sheet.ncols;
    # -hit_inds contains row index col index pairs for locations corresponding
    # to locations where the search term was relevant.
    hit_inds = list();

    for row_ii in range(rowNN):
        for col_ii in range(colNN):
            # tmp_cell_type = 1 if string and 2 if number
            tmp_cell_type = data_sheet.cell(row_ii,col_ii).ctype;
            tmp_cell_val = data_sheet.cell(row_ii,col_ii).value;
            tmp_cell_val = str(tmp_cell_val);
            #print tmp_cell_type
            #print tmp_cell_val
            if (string.lower(tmp_cell_val) == string.lower(search_str)):
                hit_inds.append([row_ii,col_ii]);
            
    return hit_inds;


def search_cells_float( data_sheet, search_str ):            
    # not necessary to define this yet.
    return hit_inds;

# this function outputs indices that are 0-indexed
def well_IDstr2index(well_id_str):
    # insert type checking

    '''
    well codes are composed of two parts 'x' + 'd' where
    'x' is in {A-H}, and 'd' is in {1-12}. dictionary
    allows translation of alphabetic character


    '''
    well_code_dict = { 'A': 0, 'B': 1, 'C': 2, 'D': 3, \
            'E': 4, 'F': 5, 'G': 6, 'H': 7 };
    # extract row and column indices
    row_ind = well_code_dict[well_id_str[0]];
    col_ind = int(well_id_str[1:])-1;
    # convert into a linear array index for storage.
    well_index = (row_ind * 12) + col_ind 
    
    return well_index

# this function assumes a 0-indexed scheme
'''
To Do:  -add error checking functionality.
'''
def well_index2IDstr(well_index, rowNN=8, colNN=12):
    IDstr_table = string.uppercase[0:rowNN];
    well_rowStr_ind = int(math.floor(well_index/colNN));
    well_rowStr = IDstr_table[well_rowStr_ind];
    well_colStr_ind = math.fmod(well_index,colNN);
    well_colStr=str(int(well_colStr_ind)+1);
    well_IDstr= well_rowStr + well_colStr;
    return well_IDstr;


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
These are found in the sds7500 "Raw Data"
'''

# no plateID recorded (ChromaCode makes one directly)
def xls_get_PlateID( xls_sheet ):
    #search_cells(
    return None;


def xls_get_BlockType( xls_sheet ):
    tmp_inds_list = search_cells( xls_sheet, 'Block Type' );
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # it is assumed that the relevant field is in the cell immediately to the right.
        block_type_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        #block_type_ctype=xls_sheet.cell(loc_inds[0],loc_inds[1]).ctype;
        return str(block_type_str);

def xls_get_Chemistry( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Chemistry' );
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # assume desired entry is in the adjacent cell to right
        chemistry_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(chemistry_str);

def xls_get_FileName( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Experiment File Name');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # assume desired entry is in the adjacent cell to right
        tmp_fname = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_fname);


def xls_get_ExperimentTime( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Experiment Run End Time');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # assume desired entry is in the adjacent cell to right
        tmp_time_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_time_str);

'''
get experiment end time.
'''
'''
1) works only for viia7
'''

# currently works only for the viia7
def xls_get_ExperimentStartTime( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Experiment Run Start Time');
    if (not tmp_inds_list):
        return None;
    else: 
        loc_inds = tmp_inds_list[0];
        tmp_time_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_time_str);

# currently works only for the viia7
def xls_get_ExperimentEndTime(xls_sheet):
    tmp_inds_list = search_cells(xls_sheet, 'Experiment Run Stop Time');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        tmp_time_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_time_str);

# currently works only for the viia7
def xls_get_Barcode(xls_sheet):
    tmp_inds_list = search_cells(xls_sheet, 'Experiment Barcode');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        barcode_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(barcode_str);


# retrieve instrument (Instrument Type)
def xls_get_InstrType( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Instrument Type');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        #assume desired entry is in the adjacent cell to right
        tmp_instr_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_instr_str);

# retrieve instrument Name (Somehow )
def xls_get_InstrName( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Instrument Name');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # assume desired entry in the adjacent cell to right
        tmp_instr_name_str = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_instr_name_str);

# retrieve instrument serial number
def xls_get_InstrSN( xls_sheet ):
    tmp_inds_list = search_cells(xls_sheet, 'Instrument Serial Number');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # assume desired entry in the adjacent cell to right
        tmp_instrSN = xls_sheet.cell(loc_inds[0], (loc_inds[1]+1)).value;
        return str(tmp_instrSN);

# retrieve instrument




# retrieve passive reference (Passive Reference) 
def xls_get_PassRef( xls_sheet):
    tmp_inds_list = search_cells(xls_sheet, 'Passive Reference');
    if (not tmp_inds_list):
        return None;
    else:
        loc_inds = tmp_inds_list[0];
        # assume desired entry is in the adjacent cell to right
        tmp_passref_str = xls_sheet.cell(loc_inds[0],(loc_inds[1]+1)).value;
        return str(tmp_passref_str);

'''
xls_getChDataCoords(xls_sheet) 

should return a data structure containing information necessary to get time-series info.
'''
#<<<<<<< HEAD
#def xls_get_sds7500_parse_info( xls_obj ):
#    xlsh_rawdata = xls_obj.sheet_by_name('Raw Data');
#    xlsh_results = xls_obj.sheet_by_name('Results');
#    xlsh_sampset = xls_obj.sheet_by_name('Sample Setup');
#    xlsh_mcdata = xls_obj.sheet_by_name('Multicomponent Data');
#    xlsh_ampdata = xls_obj.sheet_by_name('Amplification Data');
#
    #
#    tmp_parseinfo = { \
#                'Raw Data':dict(), \
#                'Results':dict(), \
#                'Sample Setup':dict(), \
#                'Multicomponent Data':dict(), \
#                'Amplification Data':dict() \
#                };


    # parse "Raw Data" excel sheet first.
    
#    tmp_inds_list = search_cells(xlsh_rawdata, 'Well');
#=======
def xls_get_sds7500_parse_info( xls_sheets ):
    xls_sheet_rawdata = xls_sheets.sheet_by_name('Raw Data');
    xls_sheet_sampsetup=xls_sheets.sheet_by_name('Sample Setup');
    xls_sheet_results=xls_sheets.sheet_by_name('Results');
    xls_sheet_mcd=xls_sheets.sheet_by_name('Multicomponent Data');
    xls_sheet_ampdata=xls_sheets.sheet_by_name('Amplification Data');
    tmp_inds_list = search_cells(xls_sheet_rawdata, 'Well');
    tmp_out_data = { \
        'well_label_coords': [0,0], \
	'well_col':0, \
	'cycle_col':1, \
	'ch1_col':2, \
	'ch2_col':3, \
	'ch3_col':4, \
	'ch4_col':5, \
	'ch5_col':6, \
	};
    if (not tmp_inds_list):
	return tmp_out_data;
    else:
        well_label_coord = tmp_inds_list[0];
        title_row = well_label_coord[0];
        # we assume that the row with data is immediately after row with titles. We can modify this to make that determination later.
        data_row = title_row + 1;
        well_col = well_label_coord[1];
        cycle_col = well_col+1;
        ch1_col = well_col+2;
        ch2_col = well_col+3; 
        ch3_col = well_col+4;
        ch4_col = well_col+5; 
        ch5_col = well_col+6;
     	
	# load results into dictionary and return the dictionary.
	tmp_out_data['well_col'] = well_col;
	tmp_out_data['cycle_col'] = cycle_col;
	tmp_out_data['ch1_col'] = ch1_col;
	tmp_out_data['ch2_col'] = ch2_col;
	tmp_out_data['ch3_col'] = ch3_col;
	tmp_out_data['ch4_col'] = ch4_col;
	tmp_out_data['ch5_col'] = ch5_col;	 
	return tmp_out_data;
'''
author:         Juhwan Yoo
args: well_tag may be a string or an index
'''
def xls_get_WellInds(xls_sheet, well_tag, rowsNN=8, colsNN=12):
    if isinstance(well_tag, str): # this means well_tag is a string
        tmp_search_str = well_tag;
    elif (isinstance(well_tag, int) | isinstance(well_tag, float) ):
        tmp_search_str = well_index2IDstr(well_tag, rowsNN, colsNN);
    else:
        return None;

    tmp_coords_list = search_cells_strip(xls_sheet, tmp_search_str);
    # replaced with search_cells_strip due to tab in cell string for viia7 files.
    #tmp_coords_list = search_cells(xls_sheet, tmp_search_str);
    row_inds = [tmp_coords_list[ii][0] for ii in range(len(tmp_coords_list))];
    col_inds = [tmp_coords_list[jj][1] for jj in range(len(tmp_coords_list))];
    ind_lists={'row_inds':row_inds, 'col_inds':col_inds};

    return ind_lists;

def xls_get_ColVals(xls_sheet, ind_list, col_index):
    col_ind_list = [xls_sheet.cell(xx ,col_index).value for xx in ind_list]; 
    return col_ind_list;

# check this function. Used to validate data extraction procedures.
def xls_get_CycVals(xls_sheet, ind_list, col_index=1, rowsNN=8,colsNN=12):
    cyc_ind_list = xls_get_ColVals(xls_sheet, ind_list, 1);
    return cyc_ind_list;

'''
dsheet_Info={'ch1_col':, 'ch2_col', 'ch3_col', 'ch4_col', 'ch5_col'};
'''
def xls_get_WellData(xls_sheet, well_tag, \
        dSheet_Info={'cyc_col':1,'ch1_col':2, 'ch2_col':3, 'ch3_col':4, \
        'ch4_col':5, 'ch5_col':6}, rowsNN=8, colsNN=12):
    #for well_ind in range(rowsNN*colsNN);
    col_ind=dSheet_Info['cyc_col'];
    ch1_col=dSheet_Info['ch1_col'];
    ch2_col=dSheet_Info['ch2_col'];
    ch3_col=dSheet_Info['ch3_col'];
    ch4_col=dSheet_Info['ch4_col'];
    ch5_col=dSheet_Info['ch5_col'];
    tmp_inds = xls_get_WellInds(xls_sheet, well_tag, rowsNN,colsNN);
    row_inds = tmp_inds['row_inds'];

    ch1_data = xls_get_ColVals(xls_sheet, row_inds, ch1_col);
    ch2_data = xls_get_ColVals(xls_sheet, row_inds, ch2_col);
    ch3_data = xls_get_ColVals(xls_sheet, row_inds, ch3_col);
    ch4_data = xls_get_ColVals(xls_sheet, row_inds, ch4_col);
    ch5_data = xls_get_ColVals(xls_sheet, row_inds, ch5_col);
    cyc_data = xls_get_ColVals(xls_sheet, row_inds, col_ind);
    # we will not insert cycle information into this data structure just yet.
    well_data_all = [ch1_data, ch2_data, ch3_data, ch4_data, ch5_data];
    well_data_all = np.array(well_data_all);
    if isinstance(well_tag, str):
        tmp_well_str = well_tag;
        tmp_well_ind = well_IDstr2index(well_tag);
    elif (isinstance(well_tag, int) | isinstance(well_tag, float)):
        tmp_well_str = well_index2IDstr(well_tag, rowsNN, colsNN);
        tmp_well_ind = well_tag;
    else:
        print "well_tag is neither a string or integer/float";
        return None;

    well_data_dict = {'well_tag':tmp_well_str, 'well_index':tmp_well_ind, \
            'ch1':np.array(ch1_data), 'ch2':np.array(ch2_data), \
            'ch3':np.array(ch3_data), 'ch4':np.array(ch4_data), \
            'ch5':np.array(ch5_data), 'cyc':np.array(cyc_data), \
            'data_array':well_data_all}

    return well_data_dict;


def xls_get_Viia7_parse_info( xls_sheet ):
    return None;


def xls_getChData( xls_sheet ):
    return None;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
These are found in the sds7500 "Sample Setup" sheet
-it does not appear to be used in ccode_7500_arajagopal_20151029_run1_data.xls
-as of Feb 24, 2016, we will not implement functionality to retrive this sheet's data
'''

def xls_get_SampleSetup( xls_obj ):
    # retrieve and file data for
    '''
    0) Sample Name          (String)
    1) Sample Color         (String)
    2) Biogroup Name        (String)
    3) Biogroup Color       (String)
    4) Target Name          (String)
    5) Target Color         (String)
    '''
    return None;
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
These are found in sfs7500 "Multicomponent Data" sheet.
'''
# This was not necessary to define
def xls_get_MCDResults(xls_obj):    #MCD stands for MulticomponentData
    # assume input of (Multicomponent Data) sheet

    # return FAM/ROX by well: add a multicomponent field

    return None;

def xls_get_MCD_FAM(xls_sheet):
    return None;

def xls_get_MCD_ROX(xls_sheet):
    return None;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
sds7500 Amplification Data sheet

Label                           sds7500                     viia7
Block Type                      'Block Type'                'Block Type'
Chemistry                       'Chemistry'                 'Chemistry'
File Name                       'Experiment File Name'      'Experiment File Name'
Experiment Run Start Time       N/A (check again)           'Experiment Run Start Time'
Experiment Run Stop Time        'Experiment Run End Time'   'Experiment Run Stop Time'
                                                            'Experiment Type'
Instrument Type                 'Instrument Type'           'Instrument Type'
Instrument Name                 N/A (check again)           'Instrument Name'
Instrument Serial Number        N/A (check again)           'Instrument Serial Number'
Passive Reference               'Passive Reference'         'Passive Reference'
Signal Smoothing On (viia7)     N/A (check again)           'Signal Smoothing On'

description:            The functions below are for importing viia7
'''

def get_viia7_BlockType(xls_sheet):
    block_type_str = xls_get_BlockType(xls_sheet);
    return block_type_str;


def xls_get_AmplificationData(xls_obj):
    return None;

def xls_get_RnStats(xls_sheet):
    return None;

def get_viia7_parse_info( xls_sheets ):
    xls_sheet_rawdata = xls_sheets.sheet_by_name('Raw Data');
    xls_sheet_sampsetup=xls_sheets.sheet_by_name('Sample Setup');
    xls_sheet_results = xls_sheets.sheet_by_name('Results');
    xls_sheet_mcd   = xls_sheets.sheet_by_name('Multicomponent Data');
    xls_sheet_ampdata=xls_sheets.sheet_by_name('Amplification Data');
    #tmp_inds_list = search_cells(xls_sheet_rawdata, 'Well');
    tmp_inds_list = search_cells(xls_sheet_rawdata, 'Well Position');
    tmp_out_data = { \
        'well_label_coords':[0,0], \
        'well_col':1, \
        'cycle_col':2, \
        'ch1_col':3, \
        'ch2_col':4, \
        'ch3_col':5, \
        'ch4_col':6, \
        'ch5_col':7, \
        'well_ind_col':0
        };
    if (not tmp_inds_list):
        return tmp_out_data;
    else:
        well_label_coord = tmp_inds_list[0];
        title_row = well_label_coord[0];
        data_row = title_row + 1;
        well_col = well_label_coord[1];
        cycle_col = well_col + 1;
        ch1_col = well_col + 2;
        ch2_col = well_col + 3;
        ch3_col = well_col + 4;
        ch4_col = well_col + 5;
        ch5_col = well_col + 6;
        # load results into dictionary and return the dictionary
        tmp_out_data['well_label_coords']=well_label_coord;
        tmp_out_data['well_col'] = well_col;
        tmp_out_data['cycle_col'] = cycle_col;
        tmp_out_data['ch1_col'] = ch1_col;
        tmp_out_data['ch2_col'] = ch2_col;
        tmp_out_data['ch3_col'] = ch3_col;
        tmp_out_data['ch4_col'] = ch4_col;
        tmp_out_data['ch5_col'] = ch5_col;

        return tmp_out_data;

def viia7_get_WellData(xls_sheet, well_tag, dSheet_Info={'cyc_col':2, 'ch1_col':3, \
    'ch2_col':4, 'ch3_col':5, 'ch4_col':6, 'ch5_col':7}, rowsNN=8, colsNN=12):
    well_data_dict = xls_get_WellData(xls_sheet, well_tag, dSheet_Info, rowsNN, colsNN);
    return well_data_dict;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
























