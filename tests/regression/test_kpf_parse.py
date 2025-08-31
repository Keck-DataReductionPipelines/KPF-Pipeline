# Test file for kpf_parse.py

from modules.Utils.kpf_parse import *

# get_datecode() method
fn = 'KP.20230708.04519.63_2D.fits'
datecode = get_datecode(fn)

# get_ObsID() method
ObsID = get_ObsID(fn)

# is_ObsID() method
thisBool = is_ObsID(ObsID)

# get_filename() method
fn = get_filename(ObsID, level='L0', fullpath=False)
fn = get_filename(ObsID, level='L0', fullpath=True)

# get_datecode_from_filename() method
datecode = get_datecode_from_filename(fn, datetime_out=True)
datecode = get_datecode_from_filename(fn, datetime_out=False)

# get_datetime_obsid() method
dt = get_datetime_obsid(ObsID)
