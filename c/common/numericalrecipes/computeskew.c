/* $Id: computeskew.c,v 1.1 2012/09/20 14:39:37 laher Exp $ */

#include <math.h>
#include "numericalrecipes.h"
#include "nanvalue.h"

double computeskew(double average, double sigma, double data[], int n)
{
    int i;
    double s = 0.0;
    double d;
    double nanvalue = NANVALUE; 

    if (sigma == 0.0) {
        return nanvalue;
    }

    for (i = 0; i < n; i++) {
        d = data[i] - average;
        s += d * d * d;
    }
    s /= (sigma * sigma * sigma * (double) n);

    return s;
}
