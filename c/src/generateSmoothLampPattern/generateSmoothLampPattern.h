#include "fitsio.h"


/* Constants. */

#define CODENAME "generateSmoothLampPattern"
#define CODEVERSION  1.0
#define DEVELOPER  "Russ Laher"

#define TERMINATE_SUCCESS 0
#define TERMINATE_WARNING 32
#define TERMINATE_FAILURE 64

#define NUM_THREADS           1         // Default value
#define	MAX_FILENAME_LENGTH 256


/*
   x is approximately along dispersion dimension.
   y is approximately along cross-dispersion dimension.
*/

#define X_WINDOW 200
#define Y_WINDOW 1
#define NSIGMA 3.0


/* Structs. */

struct arg_struct {
  int verbose;
  int tnum;
  long startindex;
  long endindex;
  long nx;
  long ny;
  int x_window;
  int y_window;
  float nsigma;
  float *masterflat;
  float *swcmi;
};


/* Prototypes. */

int printUsage(char codeName[], char version[], char developer[], int nthreads);

void timeval_print(struct timeval *tv);

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

void *compute(void *arguments);

int readHdrInfo(char imagefile[],
		int verbose,
		int *hdrNumGreen,
		long *naxis1_green,
		long *naxis2_green,
		int *nframes_green,
		char bunit_green[],
		int *hdrNumRed,
		long *naxis1_red,
		long *naxis2_red,
		int *nframes_red,
		char bunit_red[]);

int readFloatImageData(char imagefile[],
		       int verbose,
		       int hdrNumGreen,
		       int hdrNumRed,
		       float *data_green,
		       float *data_red);

int writeFloatImageData(char outFile[],
			int verbose,
			int x_window,
			int y_window,
			float nsigma,
			long naxis1_green,
			long naxis2_green,
			int nframes_green,
		        char bunit_green[],
			long naxis1_red,
			long naxis2_red,
			int nframes_red,
		        char bunit_red[],
			float *image_green,
			float *image_red);
