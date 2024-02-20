/* $Id: computejarquebera.c,v 1.1 2012/09/20 14:39:36 laher Exp $ */

#include <math.h>
#include "numericalrecipes.h"

double computejarquebera(double skew, double kurtosis, int n)
{
    double jb = (double) n / 6.0 * (skew * skew + kurtosis * kurtosis / 4.0);
    return jb;
}
