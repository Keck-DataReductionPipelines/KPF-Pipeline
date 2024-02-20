/* $Id: computeaverage.c,v 1.1 2010/12/17 20:20:09 laher Exp $ */

#include "numericalrecipes.h"

double computeaverage(double data[], int n)
{
    int i;
    double s = 0.0;

    for (i = 0; i < n; i++) {
        s += data[i];
    }
    s /= (double) n;
    return s;
}
