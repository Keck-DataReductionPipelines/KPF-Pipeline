/* $Id: computestddev.c,v 1.3 2012/09/19 18:59:12 laher Exp $ */

#include <math.h>
#include "numericalrecipes.h"

double computestddev(double average, double data[], int n)
{
    int i;
    double s = 0.0;
    double e = 0.0;
    double d;

    for (i = 0; i < n; i++) {
        d = data[i] - average;
        e += d;
        s += d * d;
    }
    if (n == 1) {
        s = (s - e * e / (double) n) / (double) n;
    } else if (n > 1) {
        s = (s - e * e / n) / (double) (n - 1);
    }
    return sqrt(s);
}
