import pytest
from dotenv import load_dotenv
from astropy.io import fits
import numpy as np
from modules.order_trace.src.alg import OrderTraceAlg
import configparser
import os
import pandas as pd
load_dotenv()


# result_data = '/Users/cwang/documents/KPF/KPF-Pipeline/modules/order_trace/tests/result_data/'
result_data = os.getenv('KPFPIPE_TEST_DATA') + '/order_trace_test/for_pytest/'

clusters_xy_fits = 'cluster_xy_info'
cluster_form_fits = 'cluster_form_info'
cluster_clean_fits = 'cluster_clean_info'
cluster_clean_border_fits = 'cluster_clean_border_info'
cluster_merge_fits = 'cluster_merge_info'
cluster_curve = 'cluster_curve'


def get_cluster_info_csv(file_path: str):
    x = None
    y = None
    index = None
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, header=None, index_col=None)
        row, col = np.shape(df.values)
        x = df.values[:, 0].astype(int) if col >= 1 else None
        y = df.values[:, 1].astype(int) if col >= 2 else None
        index = df.values[:, 2].astype(int) if col >= 3 else None
    return x, y, index


def np_equal(ary1: np.ndarray, ary2: np.ndarray):
    if np.shape(ary1) == np.shape(ary2):
        return np.all(ary1 == ary2)
    return False


def start_paras_order_trace():
    config_paras = configparser.ConfigParser()
    config_paras['PARAM'] = {
        'instrument': 'PARAS',
        'fitting_poly_degree': 3,
        'filter_par': 20,
        'locate_cluster_noise': 0.0,
        'cluster_mask': 1,
        'order_width_th': 8,
        'width_default': 7
    }

    test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    # paras_flat = '/Users/cwang/documents/KPF/KPF-Pipeline-TestData/polygon_clipping_test/paras_data/paras.flatA.fits'
    paras_flat = test_data_dir + 'polygon_clipping_test/paras_data/paras.flatA.fits'
    assert os.path.isfile(paras_flat), "paras flat doesn't exist"

    flat_data = fits.getdata(paras_flat)
    order_t = OrderTraceAlg(flat_data[1000:2200, :], config_paras)
    # order_t = OrderTraceAlg(flat_data, config_paras)
    # order_t.add_file_logger("")
    return order_t


def test_init_exceptions():
    config_paras = configparser.ConfigParser()
    config_paras['PARAM'] = {
        'instrument': 'PARAS',
        'fitting_poly_degree': 3,
        'filter_par': 20,
        'locate_cluster_noise': 0.0,
        'cluster_mask': 1,
        'order_width_th': 8,
        'width_default': 7
    }
    test_data_dir = os.getenv('KPFPIPE_TEST_DATA') + '/'
    paras_flat = test_data_dir + 'polygon_clipping_test/paras_data/paras.flatA.fits'
    assert os.path.isfile(paras_flat), "paras flat doesn't exist"
    flat_data = fits.getdata(paras_flat)

    with pytest.raises(TypeError):
        OrderTraceAlg(list())

    with pytest.raises(Exception):
        OrderTraceAlg(flat_data[20:30, 20:30])


def test_advanced_cleaning_exceptions():
    order_t = start_paras_order_trace()

    with pytest.raises(TypeError):
        order_t.advanced_cluster_cleaning_handler(list(), list(), np.array([1, 1, 1]))

    with pytest.raises(Exception):
        order_t.advanced_cluster_cleaning_handler(np.array([1, 1, 1, 2]), np.array([1, 1]), np.array([1, 2, 3, 4]))


def test_form_clusters_exceptions():
    order_t = start_paras_order_trace()

    with pytest.raises(TypeError):
        order_t.form_clusters(list(), list())

    with pytest.raises(Exception):
        order_t.form_clusters(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))


def test_locate_clusters_paras():
    order_t = start_paras_order_trace()
    inst_file = '_paras.csv'
    # test locate_clusters
    cluster_xy = order_t.locate_clusters()
    assert (isinstance(cluster_xy, dict))

    _, nx, ny,  = order_t.get_spectral_data()
    c_keys = cluster_xy.keys()
    assert 'x' in c_keys, "cluster_xy: no 'x'"
    assert 'y' in c_keys, "cluster_xy: no 'y"
    assert (cluster_xy['x'].size == cluster_xy['y'].size), "cluster_xy: unmatched x, y size"
    assert np.all((cluster_xy['x'] >= 0) & (cluster_xy['x'] < nx)), "cluster_xy: x is out of data size"
    assert np.all((cluster_xy['y'] >= 0) & (cluster_xy['y'] < ny)), "cluster_xy: y is out of data size"
    assert ('cluster_image' in c_keys and isinstance(cluster_xy['cluster_image'],
                                                     np.ndarray)), "cluster_xy: wrong image"
    assert (np.shape(cluster_xy['cluster_image']) == (ny, nx)), "cluster_xy: wrong image size"
    one_y, one_x = np.where(cluster_xy['cluster_image'] == 1)
    assert np_equal(one_y, cluster_xy['y']), "cluster_xy: x is not consistent with image pixel 1"
    assert np_equal(one_x, cluster_xy['x']), "cluster_xy: y is not consistent with image pixel 1"

    test_fits = result_data + clusters_xy_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, cluster_xy['x']), "cluster_xy: wrong x result"
        assert np_equal(t_y, cluster_xy['y']), "cluster_xy: wrong y result"
    # print("pass locate_clusters")


def test_form_clusters_paras():
    order_t = start_paras_order_trace()
    inst_file = '_paras.csv'

    in_file = result_data + clusters_xy_fits + inst_file
    assert os.path.isfile(in_file), "input file doesn't exist"

    in_x, in_y, in_index = get_cluster_info_csv(in_file)
    x, y, p_index = order_t.form_clusters(in_x, in_y)
    assert x.size == y.size, "form_clusters: unmatched x, y size"
    assert x.size == p_index.size, "form_clusters: unmatched x, index size"

    test_fits = result_data + cluster_form_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "form_cluster: wrong x result"
        assert np_equal(t_y, y), "form_cluster: wrong y result"
        assert np_equal(t_index, p_index), "form_cluster: wrong index result"
    # print("pass form_clusters")


def test_advanced_cluster_cleaning_paras():
    order_t = start_paras_order_trace()
    inst_file = '_paras.csv'

    in_file = result_data + cluster_form_fits + inst_file
    assert os.path.isfile(in_file), "input file doesn't exist"

    in_x, in_y, in_index = get_cluster_info_csv(in_file)
    x, y, p_index, all_status = order_t.advanced_cluster_cleaning_handler(in_index, in_x, in_y)
    assert x.size == y.size, "advanced_cleaning: unmatched x, y size"
    assert x.size == p_index.size, "advanced_cleaning: unmatched x, index size"

    test_fits = result_data + cluster_clean_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "advanced_cleaning_cluster: wrong x result"
        assert np_equal(t_y, y), "advanced_cleaning: wrong y result"
        assert np_equal(t_index, p_index), "advanced_cleaning: wrong index result"
    # print("pass advanced_cleaning")


def test_clean_clusters_on_borders_paras():
    order_t = start_paras_order_trace()
    inst_file = '_paras.csv'

    in_file = result_data + cluster_clean_fits + inst_file
    assert os.path.isfile(in_file), "input file doesn't exist"

    in_x, in_y, in_index = get_cluster_info_csv(in_file)
    x, y, p_index = order_t.clean_clusters_on_borders(in_x, in_y, in_index)
    assert x.size == y.size, "clean_borders: unmatched x, y size"
    assert x.size == p_index.size, "clean_borders: unmatched x, index size"

    test_fits = result_data + cluster_clean_border_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "clean_border: wrong x result"
        assert np_equal(t_y, y), "clean_border: wrong y result"
        assert np_equal(t_index, p_index), "clean_border: wrong index result"
    # print("pass clean_border")


def test_merge_clusters_and_clean_paras():
    order_t = start_paras_order_trace()
    inst_file = '_paras.csv'

    in_file = result_data + cluster_clean_border_fits + inst_file
    assert os.path.isfile(in_file), "input file doesn't exist"

    in_x, in_y, in_index = get_cluster_info_csv(in_file)
    x, y, p_index = order_t.merge_clusters_and_clean(in_index, in_x, in_y)
    assert x.size == y.size, "merge_cluster: unmatch x, y size"
    assert x.size == p_index.size, "merge_cluster: unmatch x, index size"

    test_fits = result_data + cluster_merge_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "merge_cluster: wrong x result"
        assert np_equal(t_y, y), "merge_cluster: wrong y result"
        assert np_equal(t_index, p_index), "merge_cluster: wrong index result"
    # print("pass merge_clean")


def test_find_widths_paras():
    order_t = start_paras_order_trace()
    inst_fits = '_paras.csv'

    in_file = result_data + cluster_merge_fits + inst_fits
    assert os.path.isfile(in_file), "input file doesn't exist"

    in_x, in_y, in_index = get_cluster_info_csv(in_file)
    all_widths, coeffs = order_t.find_all_cluster_widths(in_index, in_x, in_y, power_for_width_estimation=3)
    assert len(all_widths) == np.shape(coeffs)[0] - 1, "find_widths: unmatched order of widths and coeffs"
    c_df = order_t.write_cluster_info_to_dataframe(all_widths, coeffs)

    test_csv = result_data + cluster_curve+"_paras.csv"
    if os.path.isfile(test_csv):
        df = pd.read_csv(test_csv, header=None)
        assert np.all((np.absolute(df.values - c_df.values)) < 0.000001), "find_widths: unmatched fitting curves"
    # print("pass find_widths")


"""
def test_extract_order_trace_paras():
    order_t = start_paras_order_trace()
    inst_file = '_paras.csv'

    # test locate_clusters
    cluster_xy = order_t.locate_clusters()
    assert(isinstance(cluster_xy, dict))

    _, nx, ny,  = order_t.get_spectral_data()
    c_keys = cluster_xy.keys()
    assert 'x' in c_keys, "cluster_xy: no 'x'"
    assert 'y' in c_keys, "cluster_xy: no 'y"
    assert(cluster_xy['x'].size == cluster_xy['y'].size), "cluster_xy: unmatched x, y size"
    assert np.all((cluster_xy['x'] >= 0) & (cluster_xy['x'] < nx)), "cluster_xy: x is out of data size"
    assert np.all((cluster_xy['y'] >= 0) & (cluster_xy['y'] < ny)), "cluster_xy: y is out of data size"
    assert ('cluster_image' in c_keys and isinstance(cluster_xy['cluster_image'], np.ndarray)), "cluster_xy: wrong image"
    assert (np.shape(cluster_xy['cluster_image']) == (ny, nx)), "cluster_xy: wrong image size"
    one_y, one_x = np.where(cluster_xy['cluster_image'] == 1)
    assert np_equal(one_y, cluster_xy['y']), "cluster_xy: x is not consistent with image pixel 1"
    assert np_equal(one_x, cluster_xy['x']), "cluster_xy: y is not consistent with image pixel 1"

    test_fits = result_data + clusters_xy_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, cluster_xy['x']), "cluster_xy: wrong x result"
        assert np_equal(t_y, cluster_xy['y']), "cluster_xy: wrong y result"

    # print("pass locate_clusters")
    # test form_clusters
    x, y, p_index = order_t.form_clusters(cluster_xy['x'], cluster_xy['y'])
    assert x.size == y.size, "form_clusters: unmatched x, y size"
    assert x.size == p_index.size, "form_clusters: unmatched x, index size"

    test_fits = result_data + cluster_form_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "form_cluster: wrong x result"
        assert np_equal(t_y, y), "form_cluster: wrong y result"
        assert np_equal(t_index, p_index), "form_cluster: wrong index result"

    # print("pass  form_clusters")
    # test advanced cleaning
    x, y, p_index, _ = order_t.advanced_cluster_cleaning_handler(p_index, x, y)
    assert x.size == y.size, "advanced_cleaning: unmatched x, y size"
    assert x.size == p_index.size, "advanced_cleaning: unmatched x, index size"

    test_fits = result_data + cluster_clean_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "advanced_cleaning_cluster: wrong x result"
        assert np_equal(t_y, y), "advanced_cleaning: wrong y result"
        assert np_equal(t_index, p_index), "advanced_cleaning: wrong index result"

    # print("pass advanced cleaning")
    # test top and bottom borders
    x, y, p_index = order_t.clean_clusters_on_borders(x, y, p_index)
    assert x.size == y.size, "clean_borders: unmatched x, y size"
    assert x.size == p_index.size, "clean_borders: unmatched x, index size"

    test_fits = result_data + cluster_clean_border_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "clean_border: wrong x result"
        assert np_equal(t_y, y), "clean_border: wrong y result"
        assert np_equal(t_index, p_index), "clean_border: wrong index result"

    # print("pass clean_clusters_on_borders")
    # test merge clusters and remove short broken clusters
    x, y, p_index = order_t.merge_clusters_and_clean(p_index, x, y)
    assert x.size == y.size, "merge_cluster: unmatch x, y size"
    assert x.size == p_index.size, "merge_cluster: unmatch x, index size"

    test_fits = result_data + cluster_merge_fits + inst_file
    if os.path.isfile(test_fits):
        t_x, t_y, t_index = get_cluster_info_csv(test_fits)
        assert np_equal(t_x, x), "merge_cluster: wrong x result"
        assert np_equal(t_y, y), "merge_cluster: wrong y result"
        assert np_equal(t_index, p_index), "merge_cluster: wrong index result"

    # print("pass merge and clean")
    # test find widths
    all_widths, coeffs = order_t.find_all_cluster_widths(p_index, x, y, power_for_width_estimation=3)
    assert len(all_widths) == np.shape(coeffs)[0] - 1, "find_widths: unmatched order of widths and coeffs"
    c_df = order_t.write_cluster_info_to_dataframe(all_widths, coeffs)

    test_csv = result_data + cluster_curve+"_paras.csv"
    if os.path.isfile(test_csv):
        df = pd.read_csv(test_csv, header=None)
        assert np.all((df.values - c_df.values) < 0.000001), "find_widths: unmatched fitting curves"
    # print("test done")
"""