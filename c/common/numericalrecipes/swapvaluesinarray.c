/* $Id: swapvaluesinarray.c,v 1.1 2010/12/17 20:20:09 laher Exp $ */

#include "numericalrecipes.h"

void swapvaluesinarray(double d[], int i, int j)
{
    double dummy;
    dummy = d[i];
    d[i] = d[j];
    d[j] = dummy;
}
