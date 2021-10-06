#include <stdio.h>
#include <stdlib.h>

/*

CCF_3d_c.c is a C implementation of the cross correlation of a spectrum
and a mask. It is meant to be called in Python using the wrapper module
CCF_3d_cpython.py.

To run, follow the instructions given in the README to compile, then simply
call CCF_3d_cpython.calc_ccf() as you would call any other Python function.

*/

/*
 * ccf: computes the cross correlation of a spectrum and a mask.
 * 
 * arguments: 
 *  m_l: left edges of mask, length n
 *  m_h: right edges of mask, length n
 *  wav: the wavelengths of the spectrum [Angstroms], length m
 *  spec: flux values of the spectrum, length m
 *  weight: mask weights, length n
 *  sn: additional SNR scaling factor, length n (usually set to array of all 1s)
 *  v_r: the radial velocity at which to calculate the CCF [km/s]
 *  v_b: the barycentric velocity of the spectrum [km/s]
 *  n: length of mask-related arrays (see above)
 *  m: length of spectrum-related arrays (see above)
 * 
 * returns:
 *  ccf: the computed cross-correlation value
 * 
 */
double ccf(
    double m_l[], double m_h[], double wav[], double spec[], double weight[], 
    double sn[], double v_r, double v_b, int n, int m
) {

    double c = 2.99792458e5; /* Speed of light [km/s] */

    int cond;
    double gamma, ccf, snw;
    double fraction, pix_init, pix_end;

    int i, j;

    double *m_lloc;
    double *m_hloc;

    m_lloc = (double *)malloc(n * sizeof(double));
    m_hloc = (double *)malloc(n * sizeof(double));

    if ((m_lloc == NULL) | (m_hloc == NULL))
    {
        fprintf(
            stderr, "Fatal error: out of memory. Terminating program.\n"
        );
        exit(1);
    }

    /* Doppler factor, 3D. */
    gamma = (1. + (v_r / c)) / (1. + (v_b / c));

    /* Doppler shift mask; shifts all lines in the mask. */

    for (i = 0; i < n; i++) {
        m_lloc[i] = m_l[i] * gamma;
        m_hloc[i] = m_h[i] * gamma;
    }

    i = 0; /* Marks current location in mask; the mask line iterator. */
    ccf = 0.;
    snw = 0.;
    cond = 0;

    /* Loop over all wavelengths in the spectrum. */
    for (j = 1; j < m - 1; j++) {

        pix_init = 0.5 * (wav[j - 1] + wav[j]);
        pix_end = 0.5 * (wav[j] + wav[j + 1]);

        /* Loop over the mask indices. Figure out how many wavelengths there 
         * are within that pixel 
         */
        while ((m_hloc[i] < pix_init) & (cond == 0)) {
            if (i == n - 1) {
                cond = 1;
            }
            if (cond == 0) {
                i++;
            }
        }

        if ((pix_end < m_hloc[i]) & (pix_init > m_lloc[i])) {

            /* Case 1: pixel fully within mask. */
            ccf += spec[j] * weight[i] * sn[j];
            snw += sn[j] * weight[i];
        } else if (
            ((pix_end < m_hloc[i]) & (pix_init < m_lloc[i])) & 
            (pix_end > m_lloc[i])
        ) {

            /* Case 2: only right half of pixel within mask. */
            fraction = (pix_end - m_lloc[i]) / (pix_end - pix_init);
            ccf += spec[j] * weight[i] * fraction * sn[j];
            snw += fraction * sn[j] * weight[i];
        } else if (
            ((pix_end > m_hloc[i]) & (pix_init > m_lloc[i])) & 
            (pix_init < m_hloc[i])
        ) {

            /* Case 3: only left half of pixel within mask. */
            fraction = (m_hloc[i] - pix_init) / (pix_end - pix_init);
            ccf += spec[j] * weight[i] * fraction * sn[j];
            snw += fraction * sn[j] * weight[i];
        } else if ((pix_end > m_hloc[i]) & (pix_init < m_lloc[i])) {

            /* Case 4: only middle part of pixel within mask. */
            fraction = (m_hloc[i] - m_lloc[i]) / (pix_end - pix_init);
            ccf += spec[j] * weight[i] * fraction * sn[j];
            snw += fraction * sn[j] * weight[i];
        }
    }
    
    free(m_hloc);
    free(m_lloc);
    return ccf;
}

double * ccf_pixels(
    double m_l[], double m_h[], double wav[], double spec[], double weight[],
    double sn[], double v_r, double v_b, int n, int m
) {

    double c = 2.99792458e5; /* Speed of light [km/s] */

    int cond;
    double gamma, *ccf;
    double fraction, pix_init, pix_end;

    int i, j;

    double *m_lloc;
    double *m_hloc;

    m_lloc = (double *)malloc(n * sizeof(double));
    m_hloc = (double *)malloc(n * sizeof(double));

    if ((m_lloc == NULL) | (m_hloc == NULL))
    {
        fprintf(
            stderr, "Fatal error: out of memory. Terminating program.\n"
        );
        exit(1);
    }

    /* Doppler factor, 3D. */
    gamma = (1. + (v_r / c)) / (1. + (v_b / c));

    /* Doppler shift mask; shifts all lines in the mask. */

    for (i = 0; i < n; i++) {
        m_lloc[i] = m_l[i] * gamma;
        m_hloc[i] = m_h[i] * gamma;
    }

    i = 0; /* Marks current location in mask; the mask line iterator. */
    ccf =  (double *)malloc((m-2) * sizeof(double));
    cond = 0;

    /* Loop over all wavelengths in the spectrum. */
    for (j = 1; j < m - 1; j++) {

        pix_init = 0.5 * (wav[j - 1] + wav[j]);
        pix_end = 0.5 * (wav[j] + wav[j + 1]);

        ccf[j-1] = 0.0;

        /* Loop over the mask indices. Figure out how many wavelengths there
         * are within that pixel
         */
        while ((m_hloc[i] < pix_init) & (cond == 0)) {
            if (i == n - 1) {
                cond = 1;
            }
            if (cond == 0) {
                i++;
            }
        }

        if ((pix_end < m_hloc[i]) & (pix_init > m_lloc[i])) {

            /* Case 1: pixel fully within mask. */
            ccf[j-1] = spec[j] * weight[i] * sn[j];
        } else if (
            ((pix_end < m_hloc[i]) & (pix_init < m_lloc[i])) &
            (pix_end > m_lloc[i])
        ) {

            /* Case 2: only right half of pixel within mask. */
            fraction = (pix_end - m_lloc[i]) / (pix_end - pix_init);
            ccf[j-1] = spec[j] * weight[i] * fraction * sn[j];
        } else if (
            ((pix_end > m_hloc[i]) & (pix_init > m_lloc[i])) &
            (pix_init < m_hloc[i])
        ) {

            /* Case 3: only left half of pixel within mask. */
            fraction = (m_hloc[i] - pix_init) / (pix_end - pix_init);
            ccf[j-1] = spec[j] * weight[i] * fraction * sn[j];
        } else if ((pix_end > m_hloc[i]) & (pix_init < m_lloc[i])) {

            /* Case 4: only middle part of pixel within mask. */
            fraction = (m_hloc[i] - m_lloc[i]) / (pix_end - pix_init);
            ccf[j-1] = spec[j] * weight[i] * fraction * sn[j];
        }
    }

    free(m_hloc);
    free(m_lloc);
    return ccf;
}

/* 
 * Minimal test of the functonality of ccf()
 */
int main(void) {

    double ccf_output; 

    double m_l[] = {0., 1., 2.};
    double m_h[] = {0., 1., 2.};
    double weight[] = {0., 1., 2.};
    double sn[] = {0., 1., 2.};
    double wav[] = {1., 1.2, 1.3, 1.4};
    double spec[] = {1., 1.2, 1.3, 1.4};
    int n = sizeof(m_l) / sizeof(double);
    int m = sizeof(wav) / sizeof(double);

    double v_r = 4.;
    double v_b = 5.;

    ccf_output = ccf(m_l, m_h, wav, spec, weight, sn, v_r, v_b, n, m);

    return 0;
}

