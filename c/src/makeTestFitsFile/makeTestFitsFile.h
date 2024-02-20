#include "fitsio.h"

/* Constants. */

#define PROGRAM "makeTestFitsFile"
#define VERSION  "1.0"

#define TERMINATE_SUCCESS  0
#define TERMINATE_WARNING  32
#define TERMINATE_FAILURE  64


/* Structs */

typedef struct {

  char inFile[FLEN_FILENAME];
  char outFile[FLEN_FILENAME];
  double mu;
  double sig;
  unsigned long int iseed;
  int bitpix;
  long naxis1;
  long naxis2;
  int dataOption;
  int verbose;

} ParsedArguments;


/* Prototypes. */

void initializeDefaultSettings( ParsedArguments *parsedArgs );

void getCommandLineArguments( int argc, 
                              char **argv, 
                              ParsedArguments *parsedArgs );

void printUsage();

int nint( double value );

void computeImageDataStatistics_u8( int npixels,
                                    int *n, 
                                    double *muComp, 
                                    double *sigComp, 
                                    unsigned char *image_u8, 
                                    ParsedArguments *parsedArgs );

void computeImageDataStatistics_u16( int npixels,
                                     int *n, 
                                     double *muComp, 
                                     double *sigComp, 
                                     unsigned short *image_u16, 
                                     ParsedArguments *parsedArgs );

void computeImageDataStatistics_f( int npixels,
                                   int *n, 
                                   double *muComp, 
                                   double *sigComp, 
                                   float *image_f, 
                                   ParsedArguments *parsedArgs );

void loadGaussianSamplesIntoImage_u8( int npixels,
                                      double mu,
                                      double sig,
                                      unsigned char *image_u8 );

void loadGaussianSamplesIntoImage_u16( int npixels,
                                       double mu,
                                       double sig,
                                       unsigned short *image_u16 );

void loadGaussianSamplesIntoImage_f( int npixels,
                                     double mu,
                                     double sig,
                                     float *image_f );

void loadPoissonSamplesIntoImage_u8( int npixels,
                                     double mu,
                                     unsigned char *image_u8 );

void loadPoissonSamplesIntoImage_u16( int npixels,
                                      double mu,
                                      unsigned short *image_u16 );

void loadPoissonSamplesIntoImage_f( int npixels,
                                    double mu,
                                    float *image_f );
