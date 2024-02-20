/* $Id: computescale.c,v 1.2 2012/07/24 17:21:03 laher Exp $ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "numericalrecipes.h"

double computescale(double stack[], // input array of the pixels values
		    int    count    // number of elements in input array
		   ) {
        int i;
	double d, p1, p2, scale;
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

        d = 0.841;
        p2 = computepercentile(d, temp, count);

        d = 0.159;
        p1 = computepercentile(d, temp, count);

        scale = 0.5 * (p2 - p1);

        //printf("p2, p1, scale = %f\n", p2, p1, scale);

	free(temp);

	return (scale);

}

