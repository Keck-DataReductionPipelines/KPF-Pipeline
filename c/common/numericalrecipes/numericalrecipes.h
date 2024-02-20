/* $Id: numericalrecipes.h,v 1.4 2012/09/19 18:59:12 laher Exp $ */


/* Constants. */

#define VERSION  1.0

#define TERMINATE_SUCCESS  0
#define TERMINATE_WARNING  32
#define TERMINATE_FAILURE  64


/* Prototypes. */

double selectkthvalue(int k, double data[], int n);
void swapvaluesinarray(double d[], int i, int j);
double computepercentile(double d, double data[], int n);
double computesum(double data[], int n);
double computeweightedsum(double data[], double wts[], int n);
double computeaverage(double data[], int n);
double computestddev(double average, double data[], int n);
double computeskew(double average, double sigma, double data[], int n);
double computekurtosis(double average, double sigma, double data[], int n);
double computejarquebera(double skew, double kurtosis, int n);
double computemedian(double stack[], int count);
double computescale(double stack[], int count);
void computeclippedmean(double stack[], int count,  
                        double noutlier, double *sbias, 
                        double *sbiasunc, int *nsamps,  int *nrejects);
