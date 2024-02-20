/* $Id: computeclippedmean.c,v 1.4 2011/01/31 15:48:00 laher Exp $ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "numericalrecipes.h"

void computeclippedmean(double stack[], // input array of the pixels values
		        int    count, // number of elements in input array
                        double noutlier, // Number of "sigmas" in allowed-data envelope
		        double *sbias,  // the output: the super bias
                        double *sbiasunc,  // super-bias uncertainty
                        int    *nsamps,    // number of super-bias samples
                        int    *nrejects   // number of rejected samples
		       ) {
        int i;
	double d, p1, p2, median, scale, average, sigma;
	double *data = NULL;
	double *temp = NULL;

        /* Store data in temporary array since it is rearranged by function calls. */

	temp = (double *) calloc(count, sizeof(double));
	if (temp == NULL) {
		printf("ERROR: Memory not allocated successfully in numericalrecipes library for temp array.\n");
		exit(64);
	} 

	for (i = 0; i < count; i++){
	        temp[i] = stack[i];
	}

        /* Outlier rejection. */

        d = 0.5;
        median = computepercentile(d, temp, count);

        d = 0.841;
        p2 = computepercentile(d, temp, count);

        d = 0.159;
        p1 = computepercentile(d, temp, count);

        scale = 0.5 * (p2 - p1);

	data = (double *) calloc(count, sizeof(double));
	if (data == NULL) {
		printf("ERROR: Memory not allocated successfully in numericalrecipes library for data array.\n");
		exit(64);
	} 
	double lowerlimit = median - noutlier * scale;
	double upperlimit = median + noutlier * scale;
	int nvals = 0;
	int ncrs = 0;
	for (i = 0; i < count; i++){
	    if ((stack[i] > lowerlimit) && (stack[i] < upperlimit)) {
	        data[nvals++] = stack[i];
	    } else {
	        ncrs++;
	    }
	}

        average = computeaverage(data, nvals);
        sigma = computestddev(average, data, nvals);

        *sbias = average;
        *sbiasunc = sigma / sqrt((double) nvals);  // Uncertainty of the average.
        *nsamps = nvals;
        *nrejects = ncrs;

	free(temp);
	free(data);

	return;

}

