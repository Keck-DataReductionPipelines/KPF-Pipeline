/*******************************************************************************
     generateSmoothLampPattern.c     2/13/24      Russ Laher
                                                  laher@ipac.caltech.edu
                                                  California Institute of Technology
                                                  (c) 2024, All Rights Reserved.

 Generate smooth lamp patterns for master flat.

 For GREEN_CCD and RED_CCD, make a fixed lamp pattern made by computing the
 sliding-window clipped-mean image (SWCMI) of all stacked-image data from a given
 master flat (GREEN_CCD_STACK and RED_CCD_STACK FITS extensions).  The fixed
 smooth lamp pattern enables the computation elsewhere of flat-field corrections
 to remove time-evolving dust and debris signatures on the optics of the instrument
 and telescope.  A smoothing kernel 200-pixels wide (along dispersion dimension)
 by 1-pixel high (along cross-dispersion dimension) is used for computing the
 clipped mean, with 3-sigma, double-sided outlier rejection.  The kernel is
 centered on the pixel of interest.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "fitsio.h"
#include "numericalrecipes.h"
#include "nanvalue.h"
#include "generateSmoothLampPattern.h"

int debug = 0;


int main(int argc, char **argv) {

    int i, status, verbose = 0;
    char version[10];
    char codeName[FLEN_FILENAME];
    char developer[80];
    char inFile[FLEN_FILENAME];
    char outFile[FLEN_FILENAME];
    int nthreads = NUM_THREADS;
    int x_window = X_WINDOW;
    int y_window = Y_WINDOW;
    char fval[64];
    double nsigma = NSIGMA;


    /* Set code name and software version number. */

    strcpy(codeName, CODENAME);
    sprintf(version, "%.2f", CODEVERSION);
    strcpy(developer, DEVELOPER);


    /* Initialize parameters */

    status = TERMINATE_SUCCESS;


    /* Get command-line arguments. */

    strcpy(inFile, "");
    strcpy(outFile, "" );

    if (argc < 2) {
        printUsage(codeName, version, developer, nthreads);
    }

    i = 1;
    while(i < argc) {
        if (argv[i][0] == '-') {
            switch(argv[i][1]) {
                case 'h':      /* -h (help switch) */
		    printUsage(codeName, version, developer, nthreads);
                case 'i':      /* -i <input masterflat image file> */
                    if (++i >= argc) {
                        printf("-i <input masterflat image file> missing argument...\n" );
	            } else {
	                if (argv[i-1][2] == '\0')
                            sscanf(argv[i], "%s", inFile);
	            }
                    break;
                case 's':      /* -s <number of sigmas for data-allowed envelope> */
                    if (++i >= argc) {
                        printf("-s <number of sigmas for data-allowed envelope> missing argument...\n");
	            } else {
	                if (argv[i-1][2] == '\0')
  	                    sscanf(argv[i], "%s", fval);
                        nsigma = atof(fval);
	            }
                    break;
                case 'o':      /* -o <output SWCMI FITS file> */
                    if (++i >= argc) {
                        printf("-o <output SWCMI FITS file> missing argument...\n" );
	            } else {
	                if (argv[i-1][2] == '\0')
                            sscanf(argv[i], "%s", outFile);
	            }
                    break;
                case 't':      /* -t <number of processing threads> */
                    if (++i >= argc) {
                        printf("-t <number of processing threads> missing argument...\n");
	            } else {
	                if (argv[i-1][2] == '\0')
                            sscanf(argv[i], "%d", &nthreads);
	            }
                    break;
                case 'x':      /* -x <window width (pixels)> */
                    if (++i >= argc) {
                        printf("-x <window width (pixels)> missing argument...\n");
	            } else {
	                if (argv[i-1][2] == '\0')
                            sscanf(argv[i], "%d", &x_window);
	            }
                    break;
                case 'y':      /* -y <window height (pixels)> */
                    if (++i >= argc) {
                        printf("-y <window height (pixels)> missing argument...\n");
	            } else {
	                if (argv[i-1][2] == '\0')
                            sscanf(argv[i], "%d", &y_window);
	            }
                    break;
                case 'v':      /* -v (verbose switch) */
	            verbose = 1;
                    break;
                default:
                printf("Unknown argument...\n");
            }
        } else {
            printf("Command line syntax error:\n");
            printf("   Previous argument = %s\n",argv[i - 1]);
            printf("   Current argument = %s\n",argv[i]);
        }
        i++;
    }

    if (strcmp(inFile,"") == 0) {
        printf("%s %s %s\n",
	        "*** Error: No input masterflat FITS-processed-image file specified",
                "(-i <input masterflat FITS-processed-image file>);",
                "quitting...");
        exit(TERMINATE_FAILURE);
    }

    if (strcmp(outFile,"") == 0) {
        printf("%s %s %s\n",
	       "*** Error: No output SWCMI FITS file specified",
               "(-o <output SWCMI FITS file>);",
               "quitting...");
        exit(TERMINATE_FAILURE);
    }

    printf("\n%s, v. %s by %s\n\n", codeName, version, developer);
    printf("Inputs:\n");
    printf("   Input masterflat FITS filename = %s\n", inFile);
    printf("   Number of \"sigmas\" for outlier rejection = %f\n", nsigma);
    printf("   Window width (pixels) = %d\n", x_window);
    printf("   Window height (pixels) = %d\n", y_window);
    printf("   Output SWCMI FITS filename = %s\n", outFile);
    printf("   Number of processing threads = %d\n\n", nthreads);


    struct timeval tvBegin, tvEnd, tvDiff;

    // begin
    gettimeofday(&tvBegin, NULL);
    timeval_print(&tvBegin);


    /* Read in the header of the masterflat FITS file. */

    int hdrNumGreen = 0;
    long naxis1_green;
    long naxis2_green;
    int nframes_green;
    char bunit_green[64];
    int hdrNumRed = 0;
    long naxis1_red;
    long naxis2_red;
    int nframes_red;
    char bunit_red[64];

    printf( "%s\n", "Reading header info..." );
    int readhdrstatus = readHdrInfo(inFile,
				    verbose,
				    &hdrNumGreen,
				    &naxis1_green,
				    &naxis2_green,
				    &nframes_green,
				    bunit_green,
				    &hdrNumRed,
				    &naxis1_red,
				    &naxis2_red,
				    &nframes_red,
				    bunit_red);
    if (verbose > 0) {
        printf("hdrNumGreen, hdrNumRed = %d, %d\n", hdrNumGreen, hdrNumRed);
        printf("readhdrstatus, naxis1_green, naxis2_green, nframes_green, bunit_green = %d, %ld, %ld, %d, %s\n",
	       readhdrstatus, naxis1_green, naxis2_green, nframes_green, bunit_green);
        printf("readhdrstatus, naxis1_red, naxis2_red, nframes_red, bunit_red = %d, %ld, %ld, %d, %s\n",
	       readhdrstatus, naxis1_red, naxis2_red, nframes_red, bunit_red);
    }


    /* Allocate memory for input GREEN and RED stacked images. */

    long imagesize_green = naxis1_green * naxis2_green;
    float *masterflat_green;
    masterflat_green = (float *) malloc(imagesize_green * sizeof(float *));

    long imagesize_red = naxis1_red * naxis2_red;
    float *masterflat_red;
    masterflat_red = (float *) malloc(imagesize_red * sizeof(float *));


    /* Read in the masterflat image data. */

    printf( "%s\n", "Reading input masterflat image..." );
    int readmasterflatdatastatus = readFloatImageData(inFile,
						      verbose,
						      hdrNumGreen,
						      hdrNumRed,
						      masterflat_green,
						      masterflat_red);

    if (verbose > 0) {
      printf("readmasterflatdatastatus = %d\n", readmasterflatdatastatus);
    }


    /* Process the image data. */

    float *swcmi_green;
    swcmi_green = (float *) malloc(imagesize_green * sizeof(float *));

    float *swcmi_red;
    swcmi_red = (float *) malloc(imagesize_red * sizeof(float *));


    //end
    gettimeofday(&tvEnd, NULL);
    timeval_print(&tvEnd);

    // diff
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    printf("------------------------------------->Elapsed time (sec) = %ld.%06d\n", tvDiff.tv_sec, tvDiff.tv_usec);

    // begin
    gettimeofday(&tvBegin, NULL);
    timeval_print(&tvBegin);


    /* Computation for GREEN CCD */
    
    printf( "%s %d %s\n", "Computing for GREEN CCD with", nthreads, "threads..." );

    if (nthreads == 1) {

        struct arg_struct targs;
        targs.tnum = 0;
        targs.verbose = verbose;
        targs.startindex = 0;
        targs.endindex = naxis1_green - 1;
        targs.nx = naxis1_green;
        targs.ny = naxis2_green;
	targs.x_window = x_window;
	targs.y_window = y_window;
	targs.nsigma = nsigma;
        targs.masterflat = &(masterflat_green[0]);
        targs.swcmi = &(swcmi_green[0]);
        compute(&targs);
        printf( "First swcmi-image val = %f\n", swcmi_green[0] );
        int index = (targs.ny - 1) * targs.nx + targs.endindex;
        printf( "Last swcmi-image val = %f\n", swcmi_green[index] );

    } else {


        /* Initialize and set thread detached attribute */

	pthread_t threads[nthreads];
        pthread_attr_t attr;

        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


        struct arg_struct targs[nthreads];
        long sx[nthreads],ex[nthreads];

        long nsegx = naxis1_green / nthreads;
        long nremx = naxis1_green % nthreads;

        if (verbose > 0)
            printf("naxis1_green, nsegx, nremx = %ld, %ld, %ld\n", naxis1_green, nsegx, nremx);

        for (int t = 0; t < nthreads; t++) {

            long nsx = t * nsegx;
            long nex = (t + 1) * nsegx - 1;
            if (t == nthreads - 1) nex += nremx;

            if (verbose > 0) printf("t, nsx, nex = %d, %ld, %ld\n", t, nsx, nex);

            sx[t] = nsx;
            ex[t] = nex;
        }

        for (int t = 0; t < nthreads; t++) {
            targs[t].tnum = t + 1;
            targs[t].verbose = verbose;
            targs[t].startindex = sx[t];
            targs[t].endindex = ex[t];
            targs[t].nx = naxis1_green;
            targs[t].ny = naxis2_green;
	    targs[t].x_window = x_window;
	    targs[t].y_window = y_window;
	    targs[t].nsigma = nsigma;
            targs[t].masterflat = &(masterflat_green[0]);
            targs[t].swcmi = &(swcmi_green[0]);
        }


        /* Create the independent processing threads. */

        for (long t = 0; t < nthreads; t++) {
            int rc = pthread_create(&threads[t], NULL, compute, (void *) &targs[t]);
            if (rc) {
	        printf("ERROR; return code from pthread_create() of thread %ld is %d\n", t + 1, rc);
                exit(-1);
            }
        }


        /* Free attribute and wait for the other threads */

        void *threadstatus;
        pthread_attr_destroy(&attr);
        for (long t = 0; t < nthreads; t++) {
            int rc = pthread_join(threads[t], &threadstatus);
            if (rc) {
	        printf("ERROR; return code from pthread_join() for thread %ld is %d\n", t + 1, rc);
                exit(-1);
            }
	
            if (verbose > 0)
                printf("Main: completed join with thread %ld having a status of %ld\n", t + 1, (long) threadstatus);
        }

        if (verbose > 0) {
            for (long t = 0; t < nthreads; t++) {
                printf( "First swcmi-image val = %f\n", swcmi_green[t] );
                int index = (targs[t].ny - 1) * targs[t].nx + targs[t].endindex;
                printf( "-->Thread %ld: Last swcmi-image val = %f\n", t + 1, swcmi_green[index] );
            }
        }
    }


    /* Computation for RED CCD */
    
    printf( "%s %d %s\n", "Computing for RED CCD with", nthreads, "threads..." );

    if (nthreads == 1) {

        struct arg_struct targs;
        targs.tnum = 0;
        targs.verbose = verbose;
        targs.startindex = 0;
        targs.endindex = naxis1_red - 1;
        targs.nx = naxis1_red;
        targs.ny = naxis2_red;
	targs.x_window = x_window;
	targs.y_window = y_window;
	targs.nsigma = nsigma;
        targs.masterflat = &(masterflat_red[0]);
        targs.swcmi = &(swcmi_red[0]);
        compute(&targs);
        printf( "First swcmi-image val = %f\n", swcmi_red[0] );
        int index = (targs.ny - 1) * targs.nx + targs.endindex;
        printf( "Last swcmi-image val = %f\n", swcmi_red[index] );

    } else {


        /* Initialize and set thread detached attribute */

	pthread_t threads[nthreads];
        pthread_attr_t attr;

        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


        struct arg_struct targs[nthreads];
        long sx[nthreads],ex[nthreads];

        long nsegx = naxis1_red / nthreads;
        long nremx = naxis1_red % nthreads;

        if (verbose > 0)
            printf("naxis1_red, nsegx, nremx = %ld, %ld, %ld\n", naxis1_red, nsegx, nremx);

        for (int t = 0; t < nthreads; t++) {

            long nsx = t * nsegx;
            long nex = (t + 1) * nsegx - 1;
            if (t == nthreads - 1) nex += nremx;

            if (verbose > 0) printf("t, nsx, nex = %d, %ld, %ld\n", t, nsx, nex);

            sx[t] = nsx;
            ex[t] = nex;
        }

        for (int t = 0; t < nthreads; t++) {
            targs[t].tnum = t + 1;
            targs[t].verbose = verbose;
            targs[t].startindex = sx[t];
            targs[t].endindex = ex[t];
            targs[t].nx = naxis1_red;
            targs[t].ny = naxis2_red;
	    targs[t].x_window = x_window;
	    targs[t].y_window = y_window;
	    targs[t].nsigma = nsigma;
            targs[t].masterflat = &(masterflat_red[0]);
            targs[t].swcmi = &(swcmi_red[0]);
        }


        /* Create the independent processing threads. */

        for (long t = 0; t < nthreads; t++) {
            int rc = pthread_create(&threads[t], NULL, compute, (void *) &targs[t]);
            if (rc) {
	        printf("ERROR; return code from pthread_create() of thread %ld is %d\n", t + 1, rc);
                exit(-1);
            }
        }


        /* Free attribute and wait for the other threads */

        void *threadstatus;
        pthread_attr_destroy(&attr);
        for (long t = 0; t < nthreads; t++) {
            int rc = pthread_join(threads[t], &threadstatus);
            if (rc) {
	        printf("ERROR; return code from pthread_join() for thread %ld is %d\n", t + 1, rc);
                exit(-1);
            }
	
            if (verbose > 0)
                printf("Main: completed join with thread %ld having a status of %ld\n", t + 1, (long) threadstatus);
        }

        if (verbose > 0) {
            for (long t = 0; t < nthreads; t++) {
                printf( "First swcmi-image val = %f\n", swcmi_red[t] );
                int index = (targs[t].ny - 1) * targs[t].nx + targs[t].endindex;
                printf( "-->Thread %ld: Last swcmi-image val = %f\n", t + 1, swcmi_red[index] );
            }
        }
    }


    //end
    gettimeofday(&tvEnd, NULL);
    timeval_print(&tvEnd);

    // diff
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    printf("------------------------------------->Elapsed time (sec) = %ld.%06d\n", tvDiff.tv_sec, tvDiff.tv_usec);

    // begin
    gettimeofday(&tvBegin, NULL);
    timeval_print(&tvBegin);


    /* Write image data to output FITS files. */

    printf( "%s\n", "Writing output image..." );
    int writedatastatus = writeFloatImageData(outFile,
					      verbose,
					      x_window,
					      y_window,
					      nsigma,
					      naxis1_green,
					      naxis2_green,
					      nframes_green,
					      bunit_green,
					      naxis1_red,
					      naxis2_red,
					      nframes_red,
					      bunit_green,
					      swcmi_green,
					      swcmi_red);
    if (verbose > 0) {
      printf("writedatastatus = %d\n", writedatastatus);
    }


    //end
    gettimeofday(&tvEnd, NULL);
    timeval_print(&tvEnd);

    // diff
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    printf("------------------------------------->Elapsed time (sec) = %ld.%06d\n", tvDiff.tv_sec, tvDiff.tv_usec);


    /* Free memory. */

    free(masterflat_green);
    free(masterflat_red);
    free(swcmi_green);
    free(swcmi_red);


    /* Terminate properly. */

    if (nthreads == 1) {
        exit(status);
    } else {
        pthread_exit(NULL);
    }
}


void *compute(void *arguments) {

    struct arg_struct *args = arguments;

    int verbose = args->verbose;
    long startindex = args->startindex;
    long endindex = args->endindex;
    long nx = args->nx;
    long ny = args->ny;
    int x_window = args->x_window;
    int y_window = args->y_window;
    float nsigma = args->nsigma;
    float *masterflat = args->masterflat;
    float *swcmi = args->swcmi;

    if (verbose > 0) {
        printf( "%s\n", "Processing..." );
        printf("tnum, startindex, endindex, nx, ny = %d, %ld, %ld, %ld, %ld\n", args->tnum, startindex, endindex, nx, ny);
        printf("x_window, y_window, nsigma = %d, %d, %f\n", x_window, y_window, nsigma);
    }

    int x_hwin = (int) ((x_window - 1) / 2);
    int y_hwin = (int) ((y_window - 1) / 2);

    if (verbose > 0) {
        printf("x_hwin, y_hwin = %d, %d\n", x_hwin, y_hwin);
    }

    double *data = NULL;

    data = (double *) calloc(x_window * y_window, sizeof(double));

    if (data == NULL) {
        printf("%s\n",
	       "*** Error: calloc for data array failed; quitting...");
        exit(TERMINATE_FAILURE);
    }

    for (int i = startindex; i <= endindex; i++) {
        int offset = i * nx;
        for (int j = 0; j < nx; j++) {
            int index = offset + j;

            int n = 0;

            for (int ii = i - y_hwin; ii <= i + y_hwin; ii++) {
	        if (ii < 0) continue;
                if (ii >= ny) continue;
                int win_offset = ii * nx;
                for (int jj = j - x_hwin; jj <= j + x_hwin; jj++) {
	            if (jj < 0) continue;
                    if (jj >= nx) continue;
                    int win_index = win_offset + jj;

	            double dataval = masterflat[win_index];

                    if (! (dataval != 0 && iznanorinfd(dataval))) {
	                data[n] = dataval;
			n++;
	            }
	        }
            }
	
	    if (n > 0) {
                double clippedmean, clippedmeanunc;
                int nsamps, nrejects;

                computeclippedmean(data, n, nsigma, &clippedmean, &clippedmeanunc, &nsamps, &nrejects);
	
                swcmi[index] = clippedmean;

	    } else {
	
                swcmi[index] = NANVALUE;

	    }
	}
    }

    free(data);

    if (verbose > 0)
      printf("Last pixel: tnum, swcmi = %d, %f\n", args->tnum, swcmi[(ny - 1) * nx + endindex]);

    if (args->tnum > 0) pthread_exit(NULL);  // If thread number equals zero, then assume no multi-threading is called.

    return NULL;
}


/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
    long diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}


void timeval_print(struct timeval *tv) {
    char buffer[30];
    time_t curtime;

    printf("%ld.%06d", tv->tv_sec, tv->tv_usec);
    curtime = tv->tv_sec;
    strftime(buffer, 30, "%m-%d-%Y  %T", localtime(&curtime));
    printf(" = %s.%06d\n", buffer, tv->tv_usec);
}


/* Software tutorial. */

int printUsage(char codeName[], char version[], char developer[], int nthreads) {
    printf("\nCompute the sliding-window clipped-mean image (SWCMI) of an input image\n");
    printf("to generate smooth lamp patterns for KPF both GREEN and RED chips.\n");
    printf("\n%s%s %s%s%s\n\n%s\n%s\n%s\n%s\n%s\n%s\n%s %d%s\n",
           codeName,
           ", v.", version,
	   ", by ", developer,
           "Usage:",
           "-i <input masterflat FITS file>",
           "-x <window width (pixels)> (default = 200 pixels)",
           "-y <window height (pixels)> (default = 1 pixel)",
           "-s <number of \"sigmas\" for outlier rejection> (default = 3.0)",
           "-o <output SWCMI FITS file>",
           "-t <number of processing threads> (default =",
	   nthreads,
           ")\n[-v (verbose switch)]");
    exit(TERMINATE_SUCCESS);
}


/* Read in header of image. */

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
		char bunit_red[]) {

    int numHdrs = 0;
    int status = TERMINATE_SUCCESS;
    int I_fits_return_status = 0;
    fitsfile *ffp_FITS_In;
    int hdrType;
    int hdrNum;
    char CP_Comment[FLEN_COMMENT];


    /* Open input FITS file. */

    fits_open_file(&ffp_FITS_In,
	           imagefile,
	           READONLY,
       	           &I_fits_return_status);

    if (verbose > 0)
        printf( "Status after opening image file = %d\n", I_fits_return_status );
	
    if (I_fits_return_status != 0) {
        printf("%s %s%s %s\n",
       	       "*** Error: Could not open",
	       imagefile,
	       ";",
	       "quitting...");
        exit(TERMINATE_FAILURE);
    }


    /* Get number of HDUs in input FITS file. */

    fits_get_num_hdus(ffp_FITS_In,
	              &numHdrs,
       	              &I_fits_return_status );

    if (verbose == 1)
        printf("number of HDUs, status = %d, %d\n",
               numHdrs, I_fits_return_status );

    if (numHdrs < 2) {
        printf( "%s %s\n",
	        "*** Error: Input not a multi-extension FITS file;",
                "quitting..." );
        exit(TERMINATE_FAILURE);
    }


    for (int i = 1; i <= numHdrs; i++) {

        if (verbose == 1)
            printf("============> i = %d\n", i);


        /* Move to the chip's HDU. */

        hdrNum = i;
        fits_movabs_hdu( ffp_FITS_In, hdrNum , &hdrType, &I_fits_return_status );

        if (debug == 1)
            printf( "hdrNum, hdrType, status = %d, %d, %d\n",
                     hdrNum, hdrType, I_fits_return_status );


        /* Get and print number of keywords in the chip's HDU. */

        int keysexist, morekeys;

        fits_get_hdrspace(ffp_FITS_In,
                          &keysexist,
                          &morekeys,
                          &I_fits_return_status);

        if (verbose == 1)
            printf( "Number of keywords, status  = %d, %d\n",
                    keysexist, I_fits_return_status );

        for (int j = 1; j <= keysexist; j++) {

          char *CP_Keyname_Ptr, *CP_Keyvalue_Ptr;
          char card[FLEN_CARD], card_aux[FLEN_CARD];
          char CP_Keyname[FLEN_KEYWORD], CP_Keyvalue[FLEN_VALUE];
          char CP_Comment[FLEN_COMMENT];


          /* Read card from the chip's HDU of input FITS file. */

          fits_read_record( ffp_FITS_In,
                            j,
                            card,
                            &I_fits_return_status);

          if (debug > 0) {
              printf("\n\nFITS-header card:\n%s\n", card);
              printf("status = %d\n",I_fits_return_status);
          }


          /* Conditionally write card to output FITS file. */

          strcpy(card_aux, card);
          CP_Keyname_Ptr = strtok( card_aux, "=" );
          sscanf( CP_Keyname_Ptr, "%s", CP_Keyname );

          if (debug == 1)
              printf( "-------------------------------->[%s]\n", CP_Keyname );

          if (strcmp(CP_Keyname, "NAXIS1") == 0 ) {
              CP_Keyvalue_Ptr = strtok(NULL, "/");
              sscanf(CP_Keyvalue_Ptr, "%s", CP_Keyvalue);
              if (verbose == 1)
                  printf( "NAXIS1 = %s\n", CP_Keyvalue );
              long mynaxis1 = atoi( CP_Keyvalue );
              if (verbose == 1)
                  printf( "NAXIS1 = %ld\n", mynaxis1 );
          }

          if (strcmp(CP_Keyname, "NAXIS2") == 0 ) {
              CP_Keyvalue_Ptr = strtok(NULL, "/");
              sscanf(CP_Keyvalue_Ptr, "%s", CP_Keyvalue);
              if (verbose == 1)
                  printf( "NAXIS2 = %s\n", CP_Keyvalue );
                  long mynaxis2 = atoi( CP_Keyvalue );
              if (verbose == 1)
                  printf( "NAXIS2 = %ld\n", mynaxis2 );
          }


          if (strcmp(CP_Keyname, "EXTNAME") == 0 ) {
              CP_Keyvalue_Ptr = strtok(NULL, "/");
              sscanf(CP_Keyvalue_Ptr, "%s", CP_Keyvalue);


              if (verbose == 1)
                  printf( "CP_Keyvalue = %s\n", CP_Keyvalue );


              int mylen = strlen(CP_Keyvalue);

              if (verbose == 1)
                  printf( "length of CP_Keyvalue = %d\n", mylen );

              if (debug > 1) {
                  for (int jj = 0; jj < mylen; jj++) {
                      printf("jj,CP_Keyvalue[jj] = %d, %c\n", jj,CP_Keyvalue[jj]);
                  }
              }


              /* CP_Keyvalue has leading and trailing single-quote characters. */

              if (strcmp(CP_Keyvalue, "'GREEN_CCD_STACK'") == 0 ) {

                  *hdrNumGreen = i;
                  printf( "Found GREEN at hdrNum = %d\n", i );

              } else if (strcmp(CP_Keyvalue, "'RED_CCD_STACK'") == 0 ) {

                  *hdrNumRed = i;
                  printf( "Found RED at i = %d\n", i );

              }
           }
        }
    }


    /* Move to GREEN_CCD_STACK HDU. */

    hdrNum = *hdrNumGreen;
    fits_movabs_hdu( ffp_FITS_In, hdrNum , &hdrType, &I_fits_return_status );

    if (debug == 1)
        printf( "hdrNum, hdrType, status = %d, %d, %d\n",
                 hdrNum, hdrType, I_fits_return_status );


    /* Read the keywords that tell the dimensions of the data */

    long LP_naxes_green[3];
    int I_ndims_found_green;
    fits_read_keys_lng(ffp_FITS_In,
		       "NAXIS",
		       1, 3,
		       LP_naxes_green,
		       &I_ndims_found_green,
		       &I_fits_return_status);

    if (I_fits_return_status) {
        printf("%s\n",
	       "*** Error: Could not read NAXIS keywords; quitting...");
        exit(TERMINATE_FAILURE);
    }

    if (I_ndims_found_green > 2) {
        printf("%s %d %s\n",
	       "*** Error in I_ndims_found_green: A single 2-D image plane is expected; found",
	       I_ndims_found_green,
	       "image planes; quitting...");
        exit(TERMINATE_FAILURE);
    }

    *naxis1_green = (int) LP_naxes_green[0];
    *naxis2_green = (int) LP_naxes_green[1];

    if (verbose > 0)
        printf("readHdrInfo: I_ndims_found_green, naxis1_green, naxis2_green = %d, %ld, %ld\n",
	       I_ndims_found_green, *naxis1_green, *naxis2_green);

    fits_read_key(ffp_FITS_In,
                  TINT,
                  "NFRAMES",
                  nframes_green,
                  CP_Comment,
                  &I_fits_return_status);

    if (I_fits_return_status) {
        I_fits_return_status = 0;
        *nframes_green = 0;
    }

    fits_read_key(ffp_FITS_In,
                  TSTRING,
                  "BUNIT",
                  bunit_green,
                  CP_Comment,
                  &I_fits_return_status);

    if (I_fits_return_status) {
        I_fits_return_status = 0;
        strcpy(bunit_green,"Not given");
    }


    /* Move to RED_CCD_STACK HDU. */

    hdrNum = *hdrNumRed;
    fits_movabs_hdu( ffp_FITS_In, hdrNum , &hdrType, &I_fits_return_status );

    if (debug == 1)
        printf( "hdrNum, hdrType, status = %d, %d, %d\n",
                 hdrNum, hdrType, I_fits_return_status );


    /* Read the keywords that tell the dimensions of the data */

    long LP_naxes_red[3];
    int I_ndims_found_red;
    fits_read_keys_lng(ffp_FITS_In,
		       "NAXIS",
		       1, 3,
		       LP_naxes_red,
		       &I_ndims_found_red,
		       &I_fits_return_status);

    if (I_fits_return_status) {
        printf("%s\n",
	       "*** Error: Could not read NAXIS keywords; quitting...");
        exit(TERMINATE_FAILURE);
    }

    if (I_ndims_found_red > 2) {
        printf("%s %d %s\n",
	       "*** Error in I_ndims_found_red: A single 2-D image plane is expected; found",
	       I_ndims_found_red,
	       "image planes; quitting...");
        exit(TERMINATE_FAILURE);
    }

    *naxis1_red = (int) LP_naxes_red[0];
    *naxis2_red = (int) LP_naxes_red[1];

    if (verbose > 0)
        printf("readHdrInfo: I_ndims_found_red, naxis1_red, naxis2_red = %d, %ld, %ld\n",
	       I_ndims_found_red, *naxis1_red, *naxis2_red);

    fits_read_key(ffp_FITS_In,
                  TINT,
                  "NFRAMES",
                  nframes_red,
                  CP_Comment,
                  &I_fits_return_status);

    if (I_fits_return_status) {
        I_fits_return_status = 0;
        *nframes_red = 0;
    }

    fits_read_key(ffp_FITS_In,
                  TSTRING,
                  "BUNIT",
                  bunit_red,
                  CP_Comment,
                  &I_fits_return_status);

    if (I_fits_return_status) {
        I_fits_return_status = 0;
        strcpy(bunit_red,"Not given");
    }


    /* Close input FITS file. */

    fits_close_file(ffp_FITS_In, &I_fits_return_status);

    if (verbose > 0)
        printf( "readHdrInfo: Status after closing image file = %d\n", I_fits_return_status );

    if (debug > 0)
        printf("--------------------------> hdrNumGreen, hdrNumRed = %d, %d\n", *hdrNumGreen, *hdrNumRed);

    return(status);
}


/* Read in float image data. */

int readFloatImageData(char imagefile[],
		       int verbose,
		       int hdrNumGreen,
		       int hdrNumRed,
		       float *data_green,
		       float *data_red) {

    int status = TERMINATE_SUCCESS;
    int anynull;
    int I_fits_return_status = 0;
    double nullval = 0;
    fitsfile *ffp_FITS_In;
    int hdrType;
    int hdrNum;


    /* Open input FITS file. */

    fits_open_file(&ffp_FITS_In,
	           imagefile,
	           READONLY,
       	           &I_fits_return_status);

    if (verbose > 0)
        printf( "Status after opening image file = %d\n", I_fits_return_status );
	
    if (I_fits_return_status != 0) {
        printf("%s %s%s %s\n",
       	       "*** Error: Could not open",
	       imagefile,
	       ";",
	       "quitting...");
        exit(TERMINATE_FAILURE);
    }


    /* Move to the chip's GREEN_CCD_STACK HDU. */

    hdrNum = hdrNumGreen;

    fits_movabs_hdu( ffp_FITS_In, hdrNum , &hdrType, &I_fits_return_status );

    if (debug == 1)
        printf( "hdrNum, hdrType, status = %d, %d, %d\n",
                hdrNum, hdrType, I_fits_return_status );

	
    /* Read the keywords that tell the dimensions of the data */

    long LP_naxes_green[3];
    int I_ndims_found_green;
    fits_read_keys_lng(ffp_FITS_In,
		       "NAXIS",
		       1, 3,
		       LP_naxes_green,
		       &I_ndims_found_green,
		       &I_fits_return_status);

    if (I_fits_return_status) {
        printf("%s\n",
	       "*** Error: Could not read NAXIS keywords; quitting...");
        exit(TERMINATE_FAILURE);
    }

    if (I_ndims_found_green > 2) {
        printf("%s %d %s\n",
	       "*** Error in I_ndims_found_green: A single 2-D image plane is expected; found",
	       I_ndims_found_green,
	       "image planes; quitting...");
        exit(TERMINATE_FAILURE);
    }

    long naxis1_green = (long) LP_naxes_green[0];
    long naxis2_green = (long) LP_naxes_green[1];

    long imagesize_green = naxis1_green * naxis2_green;

    if (verbose > 0)
        printf("readHdrInfo: I_ndims_found_green, naxis1_green, naxis2_green = %d, %ld, %ld\n",
	       I_ndims_found_green, naxis1_green, naxis2_green);


    /* Read image. */
	
    fits_read_img(ffp_FITS_In,
		  TFLOAT,
		  1,
                  imagesize_green,
		  &nullval,
		  data_green,
		  &anynull,
		  &I_fits_return_status);

    if (verbose > 0)
        printf("Status after reading image data = %d\n", I_fits_return_status);

    if (I_fits_return_status != 0) {
        printf("%s %s%s %s\n",
       	       "*** Error: Could not read image data from",
	       imagefile,
	       ";",
	       "quitting...");
        exit(TERMINATE_FAILURE);
    }


    /* Move to the chip's RED_CCD_STACK HDU. */

    hdrNum = hdrNumRed;

    fits_movabs_hdu( ffp_FITS_In, hdrNum , &hdrType, &I_fits_return_status );

    if (debug == 1)
        printf( "hdrNum, hdrType, status = %d, %d, %d\n",
                hdrNum, hdrType, I_fits_return_status );

	
    /* Read the keywords that tell the dimensions of the data */

    long LP_naxes_red[3];
    int I_ndims_found_red;
    fits_read_keys_lng(ffp_FITS_In,
		       "NAXIS",
		       1, 3,
		       LP_naxes_red,
		       &I_ndims_found_red,
		       &I_fits_return_status);

    if (I_fits_return_status) {
        printf("%s\n",
	       "*** Error: Could not read NAXIS keywords; quitting...");
        exit(TERMINATE_FAILURE);
    }

    if (I_ndims_found_red > 2) {
        printf("%s %d %s\n",
	       "*** Error in I_ndims_found_red: A single 2-D image plane is expected; found",
	       I_ndims_found_red,
	       "image planes; quitting...");
        exit(TERMINATE_FAILURE);
    }

    long naxis1_red = (long) LP_naxes_red[0];
    long naxis2_red = (long) LP_naxes_red[1];

    long imagesize_red = naxis1_red * naxis2_red;

    if (verbose > 0)
        printf("readHdrInfo: I_ndims_found_red, naxis1_red, naxis2_red = %d, %ld, %ld\n",
	       I_ndims_found_red, naxis1_red, naxis2_red);


    /* Read image. */
	
    fits_read_img(ffp_FITS_In,
		  TFLOAT,
		  1,
                  imagesize_red,
		  &nullval,
		  data_red,
		  &anynull,
		  &I_fits_return_status);

    if (verbose > 0)
        printf("Status after reading image data = %d\n", I_fits_return_status);

    if (I_fits_return_status != 0) {
        printf("%s %s%s %s\n",
       	       "*** Error: Could not read image data from",
	       imagefile,
	       ";",
	       "quitting...");
        exit(TERMINATE_FAILURE);
    }


    /* Close input FITS file. */

    fits_close_file(ffp_FITS_In, &I_fits_return_status);

    if (verbose > 0)
        printf( "Status after closing image file = %d\n", I_fits_return_status );

    return(status);
}


/* Write float image data to output FITS file. */

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
			float *image_red) {

    int status = TERMINATE_SUCCESS;
    char CP_Keyname[FLEN_KEYWORD];
    char CP_Comment[FLEN_COMMENT];
    int I_fits_return_status = 0;
    long I_Num_Out;
    fitsfile *ffp_FITS_Out;
    int hdrType;
    int hdrNum;
    float R_Num_Out;


    /* Open output FITS file. */

    fits_create_file(&ffp_FITS_Out,
	             outFile,
       	             &I_fits_return_status);

    if (verbose > 0)
        printf("status after opening output FITS file = %d\n", I_fits_return_status);

     if (I_fits_return_status != 0) {
         if (I_fits_return_status == 105) {
             printf("%s %s %s %s\n",
                    "*** Error: Could not create",
		    outFile,
		    "(perhaps it already exists or disk quota exceeded?);",
		    "quitting...");
         } else {
             printf("%s %s%s %s\n",
		    "*** Error: Could not create",
		    outFile,
		    ";",
		    "quitting...");
        }
        exit(TERMINATE_FAILURE);
    }


    /* Create a new primary array or IMAGE extension with a specified data type and size. If the FITS file is
       currently empty then a primary array is created, otherwise a new IMAGE extension is appended to the file.
    */

    int bitpix_primary = 8;
    int naxis_primary = 0;
    long naxes_primary[0];

    fits_create_img(ffp_FITS_Out, bitpix_primary, naxis_primary, naxes_primary, &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "ORIGINSW");
    char C_String_Out_sw[64] = "generateSmoothLampPattern.c";
    sprintf(CP_Comment, "%s", "Software that created this product.");
    R_Num_Out = nsigma;	
    fits_update_key(ffp_FITS_Out,
		    TSTRING,
		    CP_Keyname,
		    C_String_Out_sw,
		    CP_Comment,
		    &I_fits_return_status);


    /* Compute and write the DATASUM and CHECKSUM keyword values for the primary header. */

    fits_write_chksum(ffp_FITS_Out, &I_fits_return_status);


    /* GREEN CCD */
    
    /* Insert a new IMAGE extension immediately following the CHDU, which is the primary HDU. */

    int bitpix_green = -32;
    int naxis_green = 2;
    long naxes_green[2];
    naxes_green[0] = naxis1_green;
    naxes_green[1] = naxis2_green;

    fits_insert_img(ffp_FITS_Out, bitpix_green, naxis_green, naxes_green, &I_fits_return_status);


    /* Move to hdrNum=2 for the GREEN_CCD_STACK HDU. */

    hdrNum = 2;

    fits_movabs_hdu(ffp_FITS_Out, hdrNum , &hdrType, &I_fits_return_status);

    if (verbose > 0)
        printf( "writeFloatImageData: hdrNum, hdrType, status = %d, %d, %d\n",
                 hdrNum, hdrType, I_fits_return_status );


    /* Write header keywords. */

    sprintf(CP_Keyname, "%s", "SIMPLE");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = 1;	
    fits_update_key(ffp_FITS_Out,
		    TLOGICAL,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "BITPIX  ");
    sprintf(CP_Comment, "%s", "IMAGE DATA TYPE");
    I_Num_Out = -32;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NAXIS");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = 2;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NAXIS1");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = naxis1_green;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NAXIS2");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = naxis2_green;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    if (nframes_green > 0) {
        sprintf(CP_Keyname, "%s", "NFRAMES");
        sprintf(CP_Comment, "%s", "Number of frames in stack");
        I_Num_Out = nframes_green;	
        fits_update_key(ffp_FITS_Out,
		        TLONG,
		        CP_Keyname,
		        &I_Num_Out,
		        CP_Comment,
		        &I_fits_return_status);
    }

    sprintf(CP_Keyname, "%s", "EXTNAME");
    sprintf(CP_Comment, "%s", "Extension name");
    char C_String_Out_green[20] = "GREEN_CCD";
    fits_update_key(ffp_FITS_Out,
		    TSTRING,
		    CP_Keyname,
		    C_String_Out_green,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "BUNIT");
    sprintf(CP_Comment, "%s", "Units of image data");
    fits_update_key(ffp_FITS_Out,
		    TSTRING,
		    CP_Keyname,
		    bunit_green,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "XWINDOW");
    sprintf(CP_Comment, "%s", "X clipped-mean kernel size (pix)");
    I_Num_Out = x_window;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "YWINDOW");
    sprintf(CP_Comment, "%s", "Y clipped-mean kernel size (pix)");
    I_Num_Out = y_window;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NSIGMA");
    sprintf(CP_Comment, "%s", "Number of sigmas for data-clipping");
    R_Num_Out = nsigma;	
    fits_update_key(ffp_FITS_Out,
		    TFLOAT,
		    CP_Keyname,
		    &R_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);


    /* Compute and write the DATASUM and CHECKSUM keyword values for the CHDU into the current header. */

    fits_write_chksum(ffp_FITS_Out, &I_fits_return_status);

    fits_flush_file(ffp_FITS_Out, &I_fits_return_status);

    if (verbose == 1)
        printf("Status after creating basic primary HDU = %d\n",
               I_fits_return_status);


    /* Output image data. */

    fits_write_img(ffp_FITS_Out,
 		   TFLOAT,
		   1,
                   naxis1_green * naxis2_green,
		   image_green,
		   &I_fits_return_status);

    if (verbose > 0)
        printf("status after writing image data = %d\n", I_fits_return_status);

    if (I_fits_return_status != 0) {
        printf("%s %s%s %s\n",
       	       "*** Error: Could not write image data to",
	       outFile,
	       ";",
	       "quitting...");
        exit(TERMINATE_FAILURE);
    }


    /* RED CCD */
    
    /* Insert a new IMAGE extension immediately following the CHDU, which is the GREEN_CCD HDU. */

    int bitpix_red = -32;
    int naxis_red = 2;
    long naxes_red[2];
    naxes_red[0] = naxis1_red;
    naxes_red[1] = naxis2_red;

    fits_insert_img(ffp_FITS_Out, bitpix_red, naxis_red, naxes_red, &I_fits_return_status);


    /* Move to hdrNum=3 for the RED_CCD_STACK HDU. */

    hdrNum = 3;

    fits_movabs_hdu(ffp_FITS_Out, hdrNum , &hdrType, &I_fits_return_status);

    if (verbose > 0)
        printf( "writeFloatImageData: hdrNum, hdrType, status = %d, %d, %d\n",
                 hdrNum, hdrType, I_fits_return_status );


    /* Write header keywords. */

    sprintf(CP_Keyname, "%s", "SIMPLE");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = 1;	
    fits_update_key(ffp_FITS_Out,
		    TLOGICAL,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "BITPIX  ");
    sprintf(CP_Comment, "%s", "IMAGE DATA TYPE");
    I_Num_Out = -32;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NAXIS");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = 2;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NAXIS1");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = naxis1_red;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NAXIS2");
    sprintf(CP_Comment, "%s", "STANDARD FITS FORMAT");
    I_Num_Out = naxis2_red;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    if (nframes_red > 0) {
        sprintf(CP_Keyname, "%s", "NFRAMES");
        sprintf(CP_Comment, "%s", "Number of frames in stack");
        I_Num_Out = nframes_red;	
        fits_update_key(ffp_FITS_Out,
		        TLONG,
		        CP_Keyname,
		        &I_Num_Out,
		        CP_Comment,
		        &I_fits_return_status);
    }

    sprintf(CP_Keyname, "%s", "EXTNAME");
    sprintf(CP_Comment, "%s", "Extension name");
    char C_String_Out_red[20] = "RED_CCD";
    fits_update_key(ffp_FITS_Out,
		    TSTRING,
		    CP_Keyname,
		    C_String_Out_red,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "BUNIT");
    sprintf(CP_Comment, "%s", "Units of image data");
    fits_update_key(ffp_FITS_Out,
		    TSTRING,
		    CP_Keyname,
		    bunit_red,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "XWINDOW");
    sprintf(CP_Comment, "%s", "X clipped-mean kernel size (pix)");
    I_Num_Out = x_window;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "YWINDOW");
    sprintf(CP_Comment, "%s", "Y clipped-mean kernel size (pix)");
    I_Num_Out = y_window;	
    fits_update_key(ffp_FITS_Out,
		    TLONG,
		    CP_Keyname,
		    &I_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);

    sprintf(CP_Keyname, "%s", "NSIGMA");
    sprintf(CP_Comment, "%s", "Number of sigmas for data-clipping");
    R_Num_Out = nsigma;	
    fits_update_key(ffp_FITS_Out,
		    TFLOAT,
		    CP_Keyname,
		    &R_Num_Out,
		    CP_Comment,
		    &I_fits_return_status);


    /* Compute and write the DATASUM and CHECKSUM keyword values for the CHDU into the current header. */

    fits_write_chksum(ffp_FITS_Out, &I_fits_return_status);

    fits_flush_file(ffp_FITS_Out, &I_fits_return_status);

    if (verbose == 1)
        printf("Status after creating basic primary HDU = %d\n",
               I_fits_return_status);


    /* Output image data. */

    fits_write_img(ffp_FITS_Out,
 		   TFLOAT,
		   1,
                   naxis1_red * naxis2_red,
		   image_red,
		   &I_fits_return_status);

    if (verbose > 0)
        printf("status after writing image data = %d\n", I_fits_return_status);

    if (I_fits_return_status != 0) {
        printf("%s %s%s %s\n",
       	       "*** Error: Could not write image data to",
	       outFile,
	       ";",
	       "quitting...");
        exit(TERMINATE_FAILURE);
    }

	
    /* Close output FITS file. */

    fits_close_file(ffp_FITS_Out, &I_fits_return_status);

    if (verbose > 0)
        printf( "Status after closing image file = %d\n", I_fits_return_status );


    return(status);
}

