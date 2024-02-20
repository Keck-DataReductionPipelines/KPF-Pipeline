/* $Id: computepercentile.c,v 1.1 2010/12/17 20:20:09 laher Exp $ */

/* E.g., input d = 0.5 to compute the median. */

#include "numericalrecipes.h"

double computepercentile(double d, double data[], int n)
{
    int k1, k2;
    double rk, p;

    rk = d * (double) (n - 1);
    k1 = (int) rk;
    k2 = k1 + 1;
    if (k2 > n - 1) k2 = n - 1;
    p = selectkthvalue(k1, data, n);
    if ( k2 != k1 ) {
        double p1, p2, slope, yint;
        p1 = p;
        p2 = selectkthvalue(k2, data, n);
        slope = (p2 - p1);
        yint = p1 - slope * (double) k1;
        p = slope * (double) rk + yint;
    }

    return p;
}
