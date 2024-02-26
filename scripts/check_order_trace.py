import pandas as pd
import numpy as np
import sys
import os
from kpfpipe.models.level0 import KPF0
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# produce png with trace on top of image from order trace result
filepath = sys.argv[1]
file = os.path.basename(filepath).replace('.fits', '')

# filedir = '/data/reference_fits/'
# file = 'kpf_20240225_master_flat'
# filepath = filedir + file + '.fits'
print(filepath)

kpfobj = KPF0.from_fits(filepath)
ccd = ['GREEN', 'RED']

power = 3
# folder to contain order trace csv
# result_csv_dir = '/data/reference_fits/'
result_csv_dir = os.path.dirname(filepath)

for color in ccd:
    plt.figure(figsize=(20, 20), frameon=False)
    ext = color + '_CCD_STACK'
    img = kpfobj[ext]
    print('min: ', np.amin(img), 'max: ', np.amax(img))
    plt.imshow(img, cmap='gray', interpolation='nearest', norm=LogNorm())
    
    ny, nx = np.shape(kpfobj[ccd[0]+'_CCD_STACK'])
    #print(ny, nx)

    ymin = 0
    ymax = ny-1
    xmin = 0
    xmax = nx-1
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    
    result_csv = result_csv_dir + '/' + file + '_' + color +'_CCD.csv'

    df = pd.read_csv(result_csv, header=0, index_col=0)
    order_trace_data = np.array(df)
    order_coeffs = np.flip(order_trace_data[:, 0:(power+1)], axis=1)
    order_trace_data[:, 0:(power+1)] = order_coeffs

    max_order = np.shape(order_trace_data)[0]
    orders = np.arange(0, max_order, dtype=int)
    
    # upper edge: cyan, lower edge: blue, trace: red
    for order in orders:
        lower_w, upper_w = order_trace_data[order, power+1], order_trace_data[order, power+2]
        x_vals = np.arange(order_trace_data[order, power+3], order_trace_data[order, power+4]+1,  dtype=int)
        #x_vals = np.arange(0, nx, dtype=int)
        y_vals = np.polyval(order_trace_data[order, 0:(power+1)], x_vals)
        y_vals_upper = y_vals + upper_w 
        y_vals_lower = y_vals - lower_w 
        w = 0.2
        plt.plot(x_vals, y_vals, 'r', linewidth=w)
        plt.plot(x_vals, y_vals_upper, 'cyan', linewidth=w)
        plt.plot(x_vals, y_vals_lower, 'b', linewidth=w)
    
    plt_dir = '/data/order_trace/plots/'
    plt.savefig(plt_dir+'order_trace_'+file+ '_'+ color+'.png', dpi=300)
    plt.savefig(plt_dir+'order_trace_'+file + '_'+color+'.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()