/* $Id: selectkthvalue.c,v 1.1 2010/12/17 20:20:09 laher Exp $ */

#include "numericalrecipes.h"

double selectkthvalue(int k, double data[], int n) 
{
    int i, j, l, mid, ir;
    double datum;

    l = 0;
    ir = n - 1;
    for (;;) {
        if (ir <= l + 1) {
            if ((ir == l + 1) && (data[ir] < data[l])) 
                swapvaluesinarray(data, l, ir);
            return data[k];
        } else {
            mid = (l + ir) >> 1;
            swapvaluesinarray(data, mid, l + 1);
            if (data[l] > data[ir]) swapvaluesinarray(data, l, ir);
            if (data[l + 1] > data[ir]) swapvaluesinarray(data, l + 1, ir);
            if (data[l] > data[l + 1]) swapvaluesinarray(data, l, l + 1);
            i = l + 1;
            j = ir;
            datum = data[l + 1];
            for (;;) {
	        do {
                    i++;
		} while (i + 1 < n && data[i] < datum);
	        do {
                    j--;
		} while (j > 0 && data[j] > datum);
	            if (j < i) break;
                    swapvaluesinarray(data, i, j);

            }
            data[l + 1] = data[j];
            data[j] = datum;
            if (j >= k) ir = j - 1;
            if (j <= k) l = i;
        }
    }
}
