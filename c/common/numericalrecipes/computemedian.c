/* $Id: computemedian.c,v 1.1 2011/01/31 15:48:00 laher Exp $ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "numericalrecipes.h"

double computemedian(double stack[], // input array of the pixels values
		   int    count
		  ) {
        int i;
	double d, median;
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

        d = 0.5;
        median = computepercentile(d, temp, count);

	free(temp);

	return (median);

}

