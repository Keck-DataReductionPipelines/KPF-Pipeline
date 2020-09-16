import pytest
from dotenv import load_dotenv
from astropy.io import fits
import numpy as np
from modules.optimal_extraction.src.alg import OptimalExtractionAlg
import configparser
import os
load_dotenv()

result_data = os.getenv('KPFPIPE_TEST_DATA') + '/optimal_extraction_test/for_pytest/paras_'
# result_data = '/Users/cwang/documents/KPF/KPF-Pipeline/modules/optimal_extraction/results/PARAS_3sigma/paras_'
rectification_method = ['optimal_norm_fraction', 'optimal_vertical_fraction', 'optimal_not_rectified']

collect_flux_fits = 'flux'
opt_extraction_fits = 'extraction'
c_order = 75


def get_flux_fits(file_path: str):
    flux, header = fits.getdata(file_path, header=True)
    w = header['NAXIS1']
    h = header['NAXIS2'] // 2
    order_data = flux[0:h, :]
    order_flat = flux[h:, :]
    return order_data, order_flat, h, w


def get_opt_fits(file_path: str):
    res = None
    if os.path.isfile(file_path):
        data, header = fits.getdata(file_path, header=True)
        res = np.reshape(data, (1, np.size(data)))
    return res


def np_equal(ary1: np.ndarray, ary2: np.ndarray, msg_prefix=""):
    if np.shape(ary1) != np.shape(ary2):
        return False, msg_prefix+'unmatched size'
    not_nan_idx_1 = np.argwhere(~np.isnan(ary1))
    not_nan_idx_2 = np.argwhere(~np.isnan(ary2))

    if np.size(not_nan_idx_1) != np.size(not_nan_idx_2):
        return False, msg_prefix+"unmatched NaN data"
    elif np.size(not_nan_idx_1) != 0:
        if not (np.array_equal(not_nan_idx_1, not_nan_idx_2)):
            return False, msg_prefix+"unmatched NaN data"

        not_nan_1 = ary1[~np.isnan(ary1)]
        not_nan_2 = ary2[~np.isnan(ary2)]
        diff_idx = np.where(np.absolute(not_nan_1 - not_nan_2) >= 0.00001)[0]
        if diff_idx.size > 0:
            return False, msg_prefix+"not equal data, total " + str(diff_idx.size) + '.'
    return True, ""


def start_paras_optimal_extraction():
    config_paras = configparser.ConfigParser()
    config_paras['PARAM'] = {
        'instrument': 'PARAS',
        'correct_method': 'sub',
        'width_default': 6
    }

    test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    paras_flat = test_data_dir + 'polygon_clipping_test/paras_data/paras.flatA.fits'
    paras_data = test_data_dir + 'polygon_clipping_test/paras_data/14feb2015/a0018.fits'
    assert os.path.isfile(paras_flat), "paras flat doesn't exist"
    assert os.path.isfile(paras_data), "paras data doesn't exist"

    order_trace_csv = test_data_dir + \
        'order_trace_test/for_optimal_extraction/paras_poly_3sigma_gaussian_pixel_3_width_3.csv'
    order_trace_result = np.genfromtxt(order_trace_csv, delimiter=',')
    order_trace_header = {'POLY DEGREE': 3}

    flat_flux = fits.open(paras_flat)
    spectrum_flux, spectrum_header = fits.getdata(paras_data, header=True)

    opt_ext_t = OptimalExtractionAlg(flat_flux[0].data, spectrum_flux, spectrum_header,
                                     order_trace_result, order_trace_header)

    return opt_ext_t


def test_init_exceptions():
    config_paras = configparser.ConfigParser()
    config_paras['PARAM'] = {
        'instrument': 'PARAS',
        'correct_method': 'sub',
        'width_default': 6
    }
    test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    paras_flat = test_data_dir + 'polygon_clipping_test/paras_data/paras.flatA.fits'
    assert os.path.isfile(paras_flat), "paras flat doesn't exist"

    paras_data = test_data_dir + 'polygon_clipping_test/paras_data/14feb2015/a0018.fits'
    assert os.path.isfile(paras_data), "praas data doesn't exist"

    order_trace_csv = test_data_dir + \
        'order_trace_test/for_optimal_extraction/paras_poly_3sigma_gaussian_pixel_3_width_3.csv'
    assert os.path.isfile(order_trace_csv), "order trace data doesn't exist"
    order_trace_result = np.genfromtxt(order_trace_csv, delimiter=',')
    order_trace_header = {'POLY DEGREE': 3}

    flat_flux = fits.open(paras_flat)
    flat_data = flat_flux[0].data

    spectrum_flux = fits.open(paras_data)
    spectrum_data = spectrum_flux[0].data
    spectrum_header = spectrum_flux[0].header

    with pytest.raises(TypeError):
        OptimalExtractionAlg(flat_flux, spectrum_data, spectrum_header, order_trace_result, order_trace_header)

    with pytest.raises(TypeError):
        OptimalExtractionAlg(flat_data, spectrum_flux, spectrum_header, order_trace_result, order_trace_header)

    with pytest.raises(TypeError):
        OptimalExtractionAlg(flat_data, spectrum_data, spectrum_header, list(order_trace_result), order_trace_header)


def test_get_flux_from_order_exceptions():
    opt_ext = start_paras_optimal_extraction()
    coeffs = opt_ext.order_coeffs[c_order]
    edges = opt_ext.get_order_edges(c_order)
    xrange = opt_ext.get_order_xrange(c_order)
    spec_flux = opt_ext.spectrum_flux
    flat_flux = opt_ext.flat_flux
    norm = OptimalExtractionAlg.VERTICAL

    with pytest.raises(Exception):
        opt_ext.get_flux_from_order(coeffs[0:3], edges, xrange, spec_flux, flat_flux, norm_direction=norm)

    with pytest.raises(Exception):
        opt_ext.get_flux_from_order(coeffs, np.array([0]), xrange, spec_flux, flat_flux, norm_direction=norm)

    with pytest.raises(Exception):
        opt_ext.get_flux_from_order(coeffs, edges, xrange, spec_flux, flat_flux[0:100, :], norm_direction=norm)

    with pytest.raises(Exception):
        opt_ext.get_flux_from_order(coeffs, edges, xrange, spec_flux, flat_flux, norm_direction=10)


def test_optimal_extraction_exceptions():
    opt_ext = start_paras_optimal_extraction()
    method = OptimalExtractionAlg.VERTICAL

    m_file = '_'+rectification_method[method]+'.fits'
    in_file = result_data + collect_flux_fits + m_file
    assert os.path.isfile(in_file), "flux collection file doesn't exist"

    order_data, order_flat, order_h, order_w = get_flux_fits(in_file)

    assert order_data is not None and \
        order_flat is not None and \
        order_h is not None and \
        order_w is not None, "some order data is missing"

    with pytest.raises(Exception):
        opt_ext.optimal_extraction(order_data[0:-1, :], order_flat, order_h, order_w)

    with pytest.raises(Exception):
        opt_ext.optimal_extraction(order_data, order_flat, order_h-1, order_w)


def get_flux_from_order_by_method(method):
    opt_ext = start_paras_optimal_extraction()

    coeffs = opt_ext.order_coeffs[c_order]
    edges = opt_ext.get_order_edges(c_order)
    xrange = opt_ext.get_order_xrange(c_order)
    spec_flux = opt_ext.spectrum_flux
    flat_flux = opt_ext.flat_flux

    order_flux = opt_ext.get_flux_from_order(coeffs, edges, xrange, spec_flux, flat_flux, norm_direction=method)
    assert 'order_data' in order_flux, 'no collected data from the order'
    assert 'order_flat' in order_flux, 'no flat data from the order'
    assert 'data_height' in order_flux, 'no data height'
    assert 'data_width' in order_flux, 'no data width'

    assert np.shape(order_flux.get('order_data')) == np.shape(order_flux.get('order_flat')), 'inconsistent data size'
    assert np.shape(order_flux.get('order_data'))[0] == order_flux.get('data_height'), 'wrong data height'
    assert np.shape(order_flux.get('order_data'))[1] == order_flux.get('data_width'), 'wrong data width'

    m_file = '_' + rectification_method[method] + '.fits'
    res_file = result_data + collect_flux_fits + m_file
    if os.path.isfile(res_file):
        order_data, order_flat, order_h, order_w = get_flux_fits(res_file)
        assert order_h == order_flux.get('data_height'), 'unmatched data height'
        assert order_w == order_flux.get('data_width'), 'unmatched data width'

        is_equal, msg = np_equal(order_data, order_flux.get('order_data'), "order_data for get_flux_from_order: ")
        assert is_equal, msg
        is_equal, msg = np_equal(order_flat, order_flux.get('order_flat'), "order_flat for get_flux_from_order: ")
        assert is_equal, msg


def test_get_flux_from_order_norect():
    method = OptimalExtractionAlg.NoRECT
    get_flux_from_order_by_method(method)


def test_get_flux_from_order_vertical():
    method = OptimalExtractionAlg.VERTICAL
    get_flux_from_order_by_method(method)


def test_get_flux_from_order_normal():
    method = OptimalExtractionAlg.NORMAL
    get_flux_from_order_by_method(method)


def optimal_extraction_by_method(method):
    m_file = '_' + rectification_method[method] + '.fits'
    in_file = result_data + collect_flux_fits + m_file
    assert os.path.isfile(in_file), "flux collection file doesn't exist"

    opt_ext = start_paras_optimal_extraction()
    order_data, order_flat, order_h, order_w = get_flux_fits(in_file)
    opt_result_obj = opt_ext.optimal_extraction(order_data, order_flat, order_h, order_w)
    assert 'extraction' in opt_result_obj, 'no optimal extraction result'

    opt_result = opt_result_obj['extraction']
    assert np.shape(opt_result)[0] == 1 and np.shape(opt_result)[1] == order_w, 'wrong result size'

    target_file = result_data + opt_extraction_fits + m_file
    if os.path.isfile(target_file):
        target_result = get_opt_fits(target_file)
        if target_result is not None:
            is_equal, msg = np_equal(target_result, opt_result, "for optimal_extraction: ")
            assert is_equal, msg + ' in_file: ' + in_file + ' target_file: ' + target_file


def test_optimal_extraction_norect():
    method = OptimalExtractionAlg.NoRECT
    optimal_extraction_by_method(method)


def test_optimal_extraction_vertical():
    method = OptimalExtractionAlg.VERTICAL
    optimal_extraction_by_method(method)


def test_optimal_extraction_normal():
    method = OptimalExtractionAlg.NORMAL
    optimal_extraction_by_method(method)
