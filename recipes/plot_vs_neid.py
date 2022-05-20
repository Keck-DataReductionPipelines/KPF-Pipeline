from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


chip = 'green'

green_echelle_orders = np.arange(103, 138)
red_echelle_orders = np.arange(71, 103)

if chip == 'green':
    echelle_orders = green_echelle_orders
elif chip == 'red':
    echelle_orders = red_echelle_orders


def echelle_to_neid_and_kpf_idx(echelle_order, chip='green'):
    if chip == 'green':
        n_green_kpf_orders = len(green_echelle_orders)
        kpf_order_index = n_green_kpf_orders - (echelle_order - green_echelle_orders[0]) - 1
    elif chip == 'red':
        n_red_kpf_orders = len(red_echelle_orders)
        kpf_order_index = n_red_kpf_orders - (echelle_order - red_echelle_orders[0]) - 1

    neid_order_idx = 70 - echelle_order + 103

    return neid_order_idx, kpf_order_index

for echelle_order in echelle_orders:
    norder, korder = echelle_to_neid_and_kpf_idx(echelle_order, chip=chip)

    try:

        # read in a KPF solar spectrum
        kpfspect = fits.open('/data/KPF-Pipeline-TestData/DRP_First_Light/KP.20220517.79981.22_L1.fits')
        kspect = 2.5 * kpfspect['{}_SCI_FLUX1'.format(chip.upper())].data[korder] / np.nanmax(kpfspect['{}_SCI_FLUX1'.format(chip.upper())].data[korder])
        
        # read in a calibrated wavelength solution
        kpfwave = fits.open('/data/KPF-Pipeline-TestData/DRP_First_Light/KP.20220517.76117.82_L1_L1_wave.fits')
        # kpfwave = fits.open('/data/KPF-Pipeline-TestData/DRP_First_Light/KP.20220505.79423.49_roughwls_green.fits')
        # kpfwave = np.append(np.zeros((1, 4080)), kpfwave['GREEN_CAL_WAVE'].data, axis=0)
        kwave = kpfwave['{}_CAL_WAVE'.format(chip.upper())].data[korder]

        # read in a NEID spectrum
        neidspect = fits.open('/data/KPF-Pipeline-TestData/DRP_First_Light/neidL2_20220310T220853.fits')
        nspect = neidspect['SCIFLUX'].data[norder] / neidspect['SCIBLAZE'].data[norder]
        nwave = neidspect['SCIWAVE'].data[norder]

        # compare
        plt.figure(figsize=(30, 5))
        plt.title('Echelle order {} (KPF {} {}; NEID {})'.format(echelle_order, chip, korder, norder))
        plt.plot(nwave, nspect, alpha=0.5, label='NEID')
        plt.plot(kwave, kspect, alpha=0.5, label='KPF')
        plt.legend()
        plt.savefig('/data/KPF-Pipeline-TestData/DRP_First_Light/LFC/order_diagnostics/order{}/order{}solar_KPF_vs_NEID.png'.format(korder, korder), dpi=250)
        plt.close()

    except:
        pass