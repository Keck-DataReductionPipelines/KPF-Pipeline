from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


green_echelle_orders = np.arange(103, 137)
red_echelle_orders = np.arange(71, 103)


def echelle_to_neid_and_kpf_idx(echelle_order, chip='green'):
    if chip == 'green':
        kpf_order_index = 35 - (echelle_order - green_echelle_orders[0]) - 1
    elif chip == 'red':
        n_red_kpf_orders = len(red_echelle_orders)
        kpf_order_index = n_red_kpf_orders - (echelle_order - red_echelle_orders[0]) - 1

    neid_order_idx = 70 - echelle_order + 103

    return neid_order_idx, kpf_order_index

for echelle_order in red_echelle_orders:
    norder, korder = echelle_to_neid_and_kpf_idx(echelle_order, chip='red')

    # read in a KPF ThAr spectrum calibrated by the DRP
    kpfspect = fits.open('/data/KPF-Pipeline-TestData/DRP_First_Light/KP.20220505.79423.49_L1_L1_wave.fits')
    kspect = kpfspect['RED_SCI_FLUX1'].data[korder]
    kwave = kpfspect['RED_CAL_WAVE'].data[korder]

    # read in a NEID spectrum
    neidspect = fits.open('/data/KPF-Pipeline-TestData/DRP_First_Light/neidL1_20220428T150506.fits')
    nspect = neidspect['CALFLUX'].data[norder]
    nwave = neidspect['CALWAVE'].data[norder]

    # compare
    plt.figure(figsize=(30, 5))
    plt.plot(nwave, nspect, alpha=0.5, label='NEID')
    plt.plot(kwave, kspect, alpha=0.5, label='KPF')
    plt.legend()
    plt.savefig('/data/KPF-Pipeline-TestData/DRP_First_Light/order_diagnostics/order{}/KPF_vs_NEID.png'.format(korder), dpi=250)
    plt.close()