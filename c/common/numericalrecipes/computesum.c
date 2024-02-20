#include "numericalrecipes.h"

double computesum(double data[], int n)
{
    int i;
    double s = 0.0;

    for (i = 0; i < n; i++) {
        s += data[i];
    }
    return s;
}
