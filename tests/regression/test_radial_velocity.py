import pytest
from dotenv import load_dotenv
from astropy.io import fits
import numpy as np
from modules.radial_velocity.src.alg import RadialVelocityAlg
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
import configparser
import os
import pandas as pd
load_dotenv()

pytest_dir = '/radial_velocity_test/for_pytest/'
result_lev2_dir = '/NEIDdata/TAUCETI_20191217/L2/'
result_data = os.getenv('KPFPIPE_TEST_DATA') + pytest_dir + 'neid_optimal_norm_fraction_023129_'

rv_fits = 'radial_velocity'
s_order = 20
e_order = 24

ratio_s_index = 10
ratio_e_index = 89


def get_result_from_rv_fits(file_path: str):
    data = None
    if os.path.isfile(file_path):
        data, header = fits.getdata(file_path, header=True)
    return data


def np_equal(ary1: np.ndarray, ary2: np.ndarray, msg_prefix=""):
    if np.shape(ary1) != np.shape(ary2):
        return False, msg_prefix+'unmatched size'
    not_nan_idx_1 = np.argwhere(~np.isnan(ary1))
    not_nan_idx_2 = np.argwhere(~np.isnan(ary2))

    if np.size(not_nan_idx_1) != np.size(not_nan_idx_2):
        return False, msg_prefix+"unmatched NaN data size "
    elif np.size(not_nan_idx_1) != 0:
        if not (np.array_equal(not_nan_idx_1, not_nan_idx_2)):
            return False, msg_prefix+"unmatched NaN data index"

        not_nan_1 = ary1[~np.isnan(ary1)]
        not_nan_2 = ary2[~np.isnan(ary2)]
        diff_idx = np.where(np.absolute(not_nan_1 - not_nan_2) >= 0.00001)[0]
        if diff_idx.size > 0:
            return False, msg_prefix+"not equal data, total " + str(diff_idx.size) + '.'
    return True, ""


def init_radial_velocity():
    config_neid = configparser.ConfigParser()
    config_neid['PARAM'] = {
        'starname': 'Tau Ceti',
        'start_rv': -20,
        'obslon': -111.600562,
        'obslat': 31.958092,
        'obsalt': 2091.0,
        'star_config_file': 'NEIDdata/TAUCETI_20191217/neid_stars.config',
        'ra': 'star/ra',
        'dec': 'star/dec',
        'pmra': 'star/pmra',
        'pmdec': 'star/pmdec',
        'parallax': 'star/plx',
        'mask': 'star/default_mask',
        'step': 0.25,
        'step_range': '[-82, 82]',
        'mask_width': 0.5,
        'air_to_vacuum': True,
        'header_date_obs': 'DATE-OBS'
    }
    rv_init = RadialVelocityAlgInit(config_neid)

    return rv_init, config_neid


def collect_data_for_rv():
    test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    assert os.path.isdir(test_data_dir), "test data directory doesn't exist"

    code = '023129'
    neid_lev1_sample = test_data_dir + 'NEIDdata/TAUCETI_20191217/L1/neidL1_20191217T' + code + '.fits'
    assert os.path.isfile(neid_lev1_sample), "NEID L1 spectrum doesn't exist"
    op_result_fits = os.getenv('KPFPIPE_TEST_DATA') + '/optimal_extraction_test/results/NEID/' + \
        'NEID_' + code + '_extraction_optimal_norm_fraction.fits'
    assert os.path.isfile(op_result_fits)

    return neid_lev1_sample, op_result_fits


def test_rv_init_exception():
    with pytest.raises(Exception):
        RadialVelocityAlgInit()


def test_rv_ccf_init_exception():
    rv_init, config_neid = init_radial_velocity()
    neid_lev1_sample, op_result_fits = collect_data_for_rv()
    order_diff = 7
    wave_hdu = 7

    neid_sample_hdulist = fits.open(neid_lev1_sample)
    wave_calib = neid_sample_hdulist[wave_hdu].data[order_diff, :]
    op_result_data, op_result_header = fits.getdata(op_result_fits, header=True)
    neid_sample_header = neid_sample_hdulist[0].header

    rv_init_res = rv_init.start()
    bad_rv_init_res = {'status': False, 'msg': 'rv init error'}

    with pytest.raises(TypeError):
        RadialVelocityAlg(list(), neid_sample_header, rv_init_res, wave_calib, config_neid)

    with pytest.raises(TypeError):
        RadialVelocityAlg(op_result_data, neid_sample_header, rv_init_res, list(), config_neid)

    with pytest.raises(Exception):
        RadialVelocityAlg(op_result_data, neid_sample_header, bad_rv_init_res, wave_calib, config_neid)


def test_compute_rv_by_cc_exception():
    rv_handler = start_neid_radial_velocity()
    rv_handler.spectro = 'none'

    with pytest.raises(Exception):
        rv_handler.compute_rv_by_cc(start_order=s_order, end_order=e_order,
                                    start_x=500, end_x=1000)


def start_neid_radial_velocity():
    rv_init, config_neid = init_radial_velocity()
    neid_lev1_sample, op_result_fits = collect_data_for_rv()

    order_diff = 7
    wave_hdu = 7

    neid_sample_hdulist = fits.open(neid_lev1_sample)
    wave_calib = neid_sample_hdulist[wave_hdu].data[order_diff:, :]
    op_result_data, op_result_header = fits.getdata(op_result_fits, header=True)
    neid_sample_header = neid_sample_hdulist[0].header

    rv_handler = RadialVelocityAlg(op_result_data, neid_sample_header,
                                   rv_init.start(), wave_calib, config_neid)

    return rv_handler


def test_neid_compute_rv_by_cc():
    rv_handler = start_neid_radial_velocity()
    _, nx, ny = rv_handler.get_spectrum()
    s_x = 600
    e_x = nx - s_x

    rv_result = rv_handler.compute_rv_by_cc(start_order=s_order, end_order=e_order, start_x=s_x, end_x=e_x)
    assert 'ccf_ary' in rv_result, "no radial velocity computation result"
    assert isinstance(rv_result['ccf_ary'], np.ndarray), "wrong radial velocity result type"

    target_file = result_data + str(s_order) + '_' + str(e_order) + '.fits'
    if os.path.isfile(target_file):
        target_data = get_result_from_rv_fits(target_file)
        if target_data is not None:
            is_equal, msg = np_equal(target_data, rv_result.get('ccf_ary'), "compute radial velocity on neid: ")
            assert is_equal, msg


def test_neid_make_reweighting_ratio_table():
    rv_file = os.getenv('KPFPIPE_TEST_DATA') + result_lev2_dir + 'neidL2_20191217T030724.fits'
    table_ref = os.getenv('KPFPIPE_TEST_DATA') + pytest_dir + 'ccf_ratio_030724_' \
                + str(ratio_s_index) + '_' + str(ratio_e_index) + '.csv'
    if os.path.exists(rv_file) and os.path.exists(table_ref):
        hdulist = fits.open(rv_file)
        ccf_data = hdulist[12].data
        ratio_df = RadialVelocityAlg.make_reweighting_ratio_table(ccf_data[ratio_s_index:ratio_e_index+1, :],
                        ratio_s_index, ratio_e_index, 'ccf_max', max_ratio=1.0)
        df_from_ref = pd.read_csv(table_ref)
        is_equal, msg = np_equal(ratio_df.values, df_from_ref.values, "not equal to the ratio table")
        assert is_equal, msg


def test_neid_reweight_ccf():
    rv_file = os.getenv('KPFPIPE_TEST_DATA') + result_lev2_dir + 'neidL2_20191217T023129.fits'
    reweighting_ratio_tbl = os.getenv('KPFPIPE_TEST_DATA') + pytest_dir + 'ccf_ratio_030724_' \
                + str(ratio_s_index) + '_' + str(ratio_e_index) + '.csv'
    reweighted_ref = os.getenv('KPFPIPE_TEST_DATA') + pytest_dir + 'reweighted_ccf_'\
                + str(ratio_s_index) + '_' + str(ratio_e_index) + '.fits'
    if os.path.exists(rv_file) and os.path.exists(reweighting_ratio_tbl) and os.path.exists(reweighted_ref):
        hdulist = fits.open(rv_file)
        ccf_data = (hdulist[12].data)[ratio_s_index:ratio_e_index+1, :]
        total_order = ratio_e_index - ratio_s_index + 1
        ratio_df = pd.read_csv(reweighting_ratio_tbl)
        reweighting_ccf = RadialVelocityAlg.reweight_ccf(ccf_data, total_order, ratio_df.values,
                                                'ccf_max', s_order=ratio_s_index,  do_analysis=True)
        reweigted_ccf_ref = fits.open(reweighted_ref)
        is_equal, msg = np_equal(reweighting_ccf, reweigted_ccf_ref[0].data, "not equal to the reweighting ref")
        assert is_equal, msg