/*******************************************************************************
  imheaders.c     1/22/13         Russ Laher 
                                  laher@ipac.caltech.edu
		                  California Institute of Technology
                                  (c) 2013, All Rights Reserved.

  List all headers in a FITS file.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "imheaders.h"
#include "fitsio.h"

/* $Id: imheaders.c,v 1.1 2013/01/24 20:53:23 laher Exp $ */

int main(int argc, char **argv) {

  fitsfile *ffp_FITS_Image;
  char version[10], card[FLEN_CARD], numstr[40];
  char inFile[FLEN_FILENAME];
  char codeName[FLEN_FILENAME];
  int i, j, status, hdunum, hdrcardsexist, morekeys, verbose = 0;
  int I_fits_return_status = 0, hdutype = 0;
  long naxes[3];


  /* Set code name and software version number. */

  strcpy(codeName, CODENAME);
  sprintf(version, "%.2f", VERSION);


  /* Initialize parameters */

  status = TERMINATE_SUCCESS;


  /* Get command-line arguments. */

  strcpy(inFile, "");

  if (argc < 2) {
    printUsage(codeName, version);
  }

  i = 1;
  while(i < argc) {
    if (argv[i][0] == '-') {
      switch(argv[i][1]) {
        case 'h':      /* -h (help switch) */
          printUsage(codeName, version);
        case 'i':      /* -i <input FITS-processed-image file> */
          if (++i >= argc) {
            printf(
              "-i <input FITS-processed-image file> missing argument...\n" );
	  } else {
	    if (argv[i-1][2] == '\0')
              sscanf(argv[i], "%s", inFile);
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

  if (! strcmp(inFile,"") != 0) {
    printf("%s %s %s\n",
	    "*** Error: No input FITS-processed-image file specified",
            "(-i <input FITS-processed-image file>);",
            "quitting...");
    exit(TERMINATE_FAILURE);
  }

  printf("\n%s, v. %s\n\n", codeName, version);
  printf("Inputs:\n");
  printf("   Input FITS file = %s\n", inFile);
  printf("   Verbose switch = %d\n\n", verbose);



  /* Open the output FITS file */


  fits_open_file(&ffp_FITS_Image,
	         inFile,
	         READONLY,
       	         &I_fits_return_status );

  fits_get_num_hdus ( ffp_FITS_Image, &hdunum, &I_fits_return_status );
  printf("Total number of HDUs in output FITS file = %d\n\n", hdunum);

  printf("Listing of header cards of all HDUs in FITS file:\n");

  for (j = 1; j <= hdunum; j++) {

    fits_movabs_hdu( ffp_FITS_Image, j, &hdutype, &I_fits_return_status );

    fits_get_hdrspace( ffp_FITS_Image,
		       &hdrcardsexist,
                       &morekeys,
                       &I_fits_return_status );

    printf("==================================================================================\n");
    printf("HDU number and type = %d and %d\n", j, hdutype);
    printf("Number of header cards in HDU = %d\n", hdrcardsexist);
    printf("==================================================================================\n");

    for (i = 1; i <= hdrcardsexist; i++) {

      fits_read_record(ffp_FITS_Image,
                       i,
                       card,
                       &I_fits_return_status );

      printf("%s\n", card);

    }
      printf("\n");
  }

  fits_close_file(ffp_FITS_Image, &I_fits_return_status);


  /* Terminate normally. */

  exit(status);
}


/* Software tutorial. */

int printUsage(char codeName[], char version[]) {
  printf("\n%s%s %s\n\n%s %s %s\n\n", 
         codeName,
         ", v.", version,
         "Usage: \n", 
         "-i <input FITS file>\n",
         "[-v (verbose switch)]");
  exit(TERMINATE_SUCCESS);
}
