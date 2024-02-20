#include "numericalrecipes.h"

double computeweightedsum(double data[], double wts[], int n)
{
    int i;
    double s = 0.0;

    for (i = 0; i < n; i++) {
        s += data[i] * wts[i];
    }
    return s;
}
