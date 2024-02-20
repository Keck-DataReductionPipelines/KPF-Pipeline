#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "numericalrecipes.h"

/* $Id: test.c,v 1.6 2012/09/20 15:41:21 laher Exp $ */

main(int argc, char **argv) {

    int n = 6;
    double data[n];
    double d;

    data[0] = 1.9;
    data[1] = 2.3;
    data[2] = 4.5;
    data[3] = 7.6;
    data[4] = 11.8;
    data[5] = 17.01;

    printf("data = ");
    int i;
    for (i = 0; i < n; i++) {
        printf("%f", data[i]);
        if (i != n - 1) printf(", ");
    }
    printf("\n"); 

    double average = computeaverage(data, n);
    double stddev = computestddev(average, data, n);
    double scale = computescale(data, n);
    double skew = computeskew(average, stddev, data, n);
    double kurt = computekurtosis(average, stddev, data, n);
    double jb = computejarquebera(skew, kurt, n);

    double median = computemedian(data, n);

    d = 0.841;
    double p2 = computepercentile(d, data, n);

    d = 0.159;
    double p1 = computepercentile(d, data, n);

    printf("average, stddev, median, p1, p2, scale, skew, kurt, jb = %f, %f, %f, %f, %f, %f, %f, %f, %f\n", 
           average, stddev, median, p1, p2, scale, skew, kurt, jb);
     
    double clippedmean, clippedmeanunc;
    int nsamps, nrejects;
    computeclippedmean(data, n, 1.5, &clippedmean, &clippedmeanunc, &nsamps, &nrejects);

    printf("clippedmean, clippedmeanunc, nsamps, nrejects = %f, %f, %d, %d\n", clippedmean, clippedmeanunc, nsamps, nrejects);     
}
