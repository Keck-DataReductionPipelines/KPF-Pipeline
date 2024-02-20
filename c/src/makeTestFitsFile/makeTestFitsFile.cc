/*******************************************************************************
     makeTestFitsFile.cc     11/10/08       Russ Laher 
                                            laher@ipac.caltech.edu
		                            California Institute of Technology
                                            (c) 2008, All Rights Reserved.

     Create a test FITS file with known statistical properties.  The mean
     and standard deviation are computed for the output image and written
     to the MEANCOMP and SIGCOMP header keywords, respectively.

     Options:
     -i Input FITS filename (REQUIRED if -d 3 is specified)
     -o Output FITS filename (REQURIED)
     -b BITPIX of output FITS file (8, 16, or -32 allowed; default = -32)
     -d Data distribution (REQURIED)
        1 = Randomly sampled from Gaussion distribution
        2 = Randomly sampled from Poisson distribution
        3 = Read from existing FITS file, whose NAXIS# values must
            be same as those specified by the -n and -m options,
            and pass it through to the output FITS file 
     -e Random-number-generator seed
     -n NAXIS1 of output FITS file (REQURIED)
     -m NAXIS2 of output FITS file (REQURIED)
     -a Mean of data distribution (default = 0.0)
     -s Standard deviation of data distribution (default = 1.0)

     Switches:
      -h (help)
      -v (verbose)

     Required libraries: cfitsio, gsl, nanvalue;

*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "makeTestFitsFile.h"
#include "nanvalue.h"

#include "fitsio.h"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

gsl_rng * rng; /* global generator */ 

int main( int argc, char **argv ) {

  unsigned char *image_u8 = NULL;
  unsigned short *image_u16 = NULL;
  int i, n, ivalue;
  int I_fits_return_status = 0, I_dims_found, naxis, npixels;
  int anynull;
  long axes[3], naxes[2];
  long in_naxis1 = 0, in_naxis2 = 0;
  char CP_Keyname[FLEN_KEYWORD], CP_Keyvalue[FLEN_VALUE];
  char CP_Comment[FLEN_COMMENT];
  char *CP_Keyname_Ptr, *CP_Keyvalue_Ptr;
  float *image_f = NULL;
  double nullval = 0, rvalue, muComp, sigComp;
  double *in_image = NULL;
  fitsfile *ffp_FITS_In, *ffp_FITS_Out;

  const gsl_rng_type * T;
 
  ParsedArguments parsedArgs;


  /* Initialize default settings. */

  initializeDefaultSettings( &parsedArgs );


  /* Get command-line arguments. */

  getCommandLineArguments( argc, argv, &parsedArgs );
  npixels = parsedArgs.naxis1 * parsedArgs.naxis2;


  /* Allocate output image-data array. */

  if ( parsedArgs.bitpix == 8 ) {
    image_u8 = (unsigned char *) calloc( npixels, sizeof(unsigned char));
    if ( image_u8 == NULL ) {
      printf( "*** Error: output image_u8 data-array allocation failed" );
    }
  } else if ( parsedArgs.bitpix == 16 ) {
    image_u16 = (unsigned short *) calloc( npixels, sizeof(unsigned short));
    if ( image_u16 == NULL ) {
      printf( "*** Error: output image_u16 data-array allocation failed" );
    }
  } else {
    image_f = (float *) calloc( npixels, sizeof(float));
    if ( image_f == NULL ) {
      printf( "*** Error: output image_f data-array allocation failed" );
    }
  }


  /* Set up random-number generator. */

  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  rng = gsl_rng_alloc (T); 
  gsl_rng_set( rng, parsedArgs.iseed); 
  printf ("generator type: %s\n", gsl_rng_name (rng)); 
  printf ("first value = %lu\n", gsl_rng_get (rng)); 


  /* Load image-data array */

  if ( parsedArgs.dataOption == 1 ) {


    /* ...with random sampling from a Gaussian distribution. */

    if ( parsedArgs.bitpix == 8 ) {
      loadGaussianSamplesIntoImage_u8( npixels,
                                       parsedArgs.mu,
                                       parsedArgs.sig,
                                       image_u8 );    
    } else if ( parsedArgs.bitpix == 16 ) {
      loadGaussianSamplesIntoImage_u16( npixels,
                                        parsedArgs.mu,
                                        parsedArgs.sig,
                                        image_u16 );    
    } else {
      loadGaussianSamplesIntoImage_f( npixels,
                                      parsedArgs.mu,
                                      parsedArgs.sig,
                                      image_f );
    }

  } else if ( parsedArgs.dataOption == 2 ) {


    /* ...with random sampling from a Poisson distribution. */

    if ( parsedArgs.bitpix == 8 ) {
      loadPoissonSamplesIntoImage_u8( npixels,
                                      parsedArgs.mu,
                                      image_u8 );    
    } else if ( parsedArgs.bitpix == 16 ) {
      loadPoissonSamplesIntoImage_u16( npixels,
                                       parsedArgs.mu,
                                       image_u16 );    
    } else {
      loadPoissonSamplesIntoImage_f( npixels,
                                     parsedArgs.mu,
                                     image_f );
    }

  } else {


    /* ...with image data read from the input FITS file. */

    /* Open input FITS file. */

    fits_open_file( &ffp_FITS_In,
	            parsedArgs.inFile,
                    READONLY,
       	            &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_open_file) status = %d\n", 
        I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }


    /* Read NAXIS# header keywords. */

    sprintf(CP_Keyname, "%s", "NAXIS");
    fits_read_keys_lng( ffp_FITS_In,
		        CP_Keyname,
		        1, 3,
		        axes,
		        &I_dims_found,
		        &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_read_keys_lng) status = %d\n", 
        I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }
    if ( axes[2] ==0 || I_dims_found==2 ){
      axes[2] = 1;
    }
    if ( parsedArgs.naxis1 != axes[0] ) {
      printf( "*** Error: NAXIS1 must be same as input image's = %d\n", 
        I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }
    if ( parsedArgs.naxis2 != axes[1] ) {
      printf( "*** Error: NAXIS2 must be same as input image's = %d\n", 
        I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }


    /* Allocate input image-data array. */

    in_image = (double *) calloc( axes[0] * axes[1] * axes[2], sizeof(double));
    if ( in_image == NULL ) {
      printf( "*** Error: input image data-array allocation failed" );
    }


    /* Read input image data. */

    fits_read_img( ffp_FITS_In,
		   TDOUBLE,
		   (long) 1, (long) axes[0] * axes[1] * axes[2],
		   &nullval,
		   in_image,
		   &anynull,
		   &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_read_img) status = %d\n", 
        I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }

    if ( parsedArgs.bitpix == 8 ) {
      for (i = 0; i < npixels; i++) {
        if ( in_image[i] < 0.0 ) { in_image[i] = 0.0; }
        if ( in_image[i] > 255.0 ) { in_image[i] = 255.0; }
        image_u8[i] = (unsigned char) nint( in_image[i] );
      }
    } else if ( parsedArgs.bitpix == 16 ) {
      for (i = 0; i < npixels; i++) {
        if ( in_image[i] < 0.0 ) { in_image[i] = 0.0; }
        if ( in_image[i] > 65535.0 ) { in_image[i] = 65535.0; }
        image_u16[i] = (unsigned short) nint( in_image[i] );
      }
    } else {
      for (i = 0; i < npixels; i++) {
        image_f[i] = (float) nint( in_image[i] );
      }
    }

  }


  /* Compute mean and sample standard deviation of image data. */

  if ( parsedArgs.bitpix == 8 ) {
    computeImageDataStatistics_u8( npixels,
                                   &n, 
                                   &muComp, 
                                   &sigComp, 
                                   image_u8, 
                                   &parsedArgs );
  } else if ( parsedArgs.bitpix == 16 ) {
    computeImageDataStatistics_u16( npixels,
                                    &n, 
                                    &muComp, 
                                    &sigComp, 
                                    image_u16, 
                                    &parsedArgs );
  } else {
    computeImageDataStatistics_f( npixels,
                                  &n, 
                                  &muComp, 
                                  &sigComp, 
                                  image_f, 
                                  &parsedArgs );
  }


  /* Open output FITS file. */

  fits_create_file( &ffp_FITS_Out,
	          parsedArgs.outFile,
       	          &I_fits_return_status );

  if ( I_fits_return_status != 0 ) {
    printf( "*** Error: (fits_create_file) status = %d\n", I_fits_return_status );
    exit( TERMINATE_FAILURE );
  }

  naxis = 2;
  naxes[0] = parsedArgs.naxis1;
  naxes[1] = parsedArgs.naxis2;

  if ( parsedArgs.bitpix == 8 ) {

    fits_create_img( ffp_FITS_Out,
  		     BYTE_IMG,
		     naxis, 
                     naxes,
		     &I_fits_return_status );

  } else if ( parsedArgs.bitpix == 16 ) {

    fits_create_img( ffp_FITS_Out,
  		     USHORT_IMG,
		     naxis, 
                     naxes,
		     &I_fits_return_status );

  } else {

    fits_create_img( ffp_FITS_Out,
  		     FLOAT_IMG,
		     naxis, 
                     naxes,
		     &I_fits_return_status );


  }

  /* Write header keywords to FITS file. */


  sprintf( CP_Comment, "FITS image created by %s%s%s", 
    PROGRAM, ", v. ", VERSION );
  fits_write_comment( ffp_FITS_Out,
		      CP_Comment,
		      &I_fits_return_status );

  if ( I_fits_return_status != 0 ) {
    printf( "*** Error: (fits_write_comment) status = %d\n", 
      I_fits_return_status );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs.dataOption != 3 ) {

    if ( parsedArgs.dataOption == 1 ) {

      sprintf( CP_Comment, 
        "Image data randomly sampled from a Gaussian distribution" );
      fits_write_comment( ffp_FITS_Out,
		          CP_Comment,
		          &I_fits_return_status );

      if ( I_fits_return_status != 0 ) {
        printf( "*** Error: (fits_write_comment) status = %d\n", 
          I_fits_return_status );
        exit( TERMINATE_FAILURE );
      }

    } else {

      sprintf( CP_Comment, 
        "Image data randomly sampled from a Poisson distribution" );
      fits_write_comment( ffp_FITS_Out,
		          CP_Comment,
		          &I_fits_return_status );

      if ( I_fits_return_status != 0 ) {
        printf( "*** Error: (fits_write_comment) status = %d\n", 
          I_fits_return_status );
        exit( TERMINATE_FAILURE );
      }

    }

    sprintf( CP_Keyname, "RNDSEED" );
    sprintf( CP_Comment, "Random-number-generator seed" );
    ivalue = parsedArgs.iseed;
    fits_write_key( ffp_FITS_Out,
		    TINT,
		    CP_Keyname,
		    &ivalue,
		    CP_Comment,
		    &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_write_key: %s) status = %d\n", 
        CP_Keyname, I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }

    sprintf( CP_Keyname, "MEANREQ" );
    sprintf( CP_Comment, "Mean specified for image data" );
    rvalue = parsedArgs.mu;
    fits_write_key( ffp_FITS_Out,
		    TDOUBLE,
		    CP_Keyname,
		    &rvalue,
		    CP_Comment,
		    &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_write_key: %s) status = %d\n", 
        CP_Keyname, I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }

    sprintf( CP_Keyname, "SIGREQ" );
    sprintf( CP_Comment, "Sigma specified for image data" );
    rvalue = parsedArgs.sig;
    fits_write_key( ffp_FITS_Out,
		    TDOUBLE,
		    CP_Keyname,
		    &rvalue,
		    CP_Comment,
		    &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_write_key: %s) status = %d\n", 
        CP_Keyname, I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }

  } else {

    sprintf( CP_Comment, 
      "Image data read from a FITS file" );
    fits_write_comment( ffp_FITS_Out,
	                CP_Comment,
	                &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_write_comment) status = %d\n", 
        I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }

    sprintf( CP_Keyname, "INPFILE" );
    sprintf( CP_Keyvalue, "%s", parsedArgs.inFile );
    sprintf( CP_Comment, "Input-file source for image data" );
    fits_write_key( ffp_FITS_Out,
		    TSTRING,
		    CP_Keyname,
		    &CP_Keyvalue,
		    CP_Comment,
		    &I_fits_return_status );

    if ( I_fits_return_status != 0 ) {
      printf( "*** Error: (fits_write_key: %s) status = %d\n", 
        CP_Keyname, I_fits_return_status );
      exit( TERMINATE_FAILURE );
    }

  }

  sprintf( CP_Keyname, "MEANCOMP" );
  sprintf( CP_Comment, "Mean computed from image data" );
  rvalue = muComp;
  fits_write_key( ffp_FITS_Out,
		  TDOUBLE,
		  CP_Keyname,
		  &rvalue,
		  CP_Comment,
		  &I_fits_return_status );

  if ( I_fits_return_status != 0 ) {
    printf( "*** Error: (fits_write_key: %s) status = %d\n", 
      CP_Keyname, I_fits_return_status );
    exit( TERMINATE_FAILURE );
  }

  sprintf( CP_Keyname, "SIGCOMP" );
  sprintf( CP_Comment, "Sample sigma computed from image data" );
  rvalue = sigComp;
  fits_write_key( ffp_FITS_Out,
		  TDOUBLE,
		  CP_Keyname,
		  &rvalue,
		  CP_Comment,
		  &I_fits_return_status );

  if ( I_fits_return_status != 0 ) {
    printf( "*** Error: (fits_write_key: %s) status = %d\n", 
      CP_Keyname, I_fits_return_status );
    exit( TERMINATE_FAILURE );
  }


  /* Write image data to FITS file. */

  if ( parsedArgs.bitpix == 8 ) {

    fits_write_img( ffp_FITS_Out,
		    TBYTE,
		    1, 
                    npixels,
		    image_u8,
		    &I_fits_return_status );

  } else if ( parsedArgs.bitpix == 16 ) {

    fits_write_img( ffp_FITS_Out,
		    TUSHORT,
		    1, 
                    npixels,
		    image_u16,
		    &I_fits_return_status );

  } else {		

    fits_write_img( ffp_FITS_Out,
		    TFLOAT,
		    1, 
                    npixels,
		    image_f,
		    &I_fits_return_status );

  }		

  if ( I_fits_return_status != 0 ) {
    printf( "*** Error: (fits_write_img) status = %d\n", I_fits_return_status );
    exit( TERMINATE_FAILURE );
  }


  /* Close input FITS file. */

  fits_close_file( ffp_FITS_Out, &I_fits_return_status );

  if ( I_fits_return_status != 0 ) {
    printf( "*** Error: (fits_close_file) status = %d\n", I_fits_return_status );
    exit( TERMINATE_FAILURE );
  }

  exit( TERMINATE_SUCCESS );
}


void initializeDefaultSettings( ParsedArguments *parsedArgs )
{
  parsedArgs->mu = 0.0;
  parsedArgs->sig = 1.0;
  parsedArgs->iseed = 999999999;
  parsedArgs->bitpix = -32;
  parsedArgs->naxis1 = 0; 
  parsedArgs->naxis2 = 0;
  parsedArgs->dataOption = 0;
  parsedArgs->verbose = 0;
}

void getCommandLineArguments( int argc, 
                              char **argv, 
                              ParsedArguments *parsedArgs )
{
  int i;

  strcpy( parsedArgs->outFile, "" );
  strcpy( parsedArgs->inFile, "" );

  if (argc < 2) {
    printUsage();
  }

  i = 1;
  while( i < argc ) {
    if ( argv[i][0] == '-' ) {
      switch( argv[i][1] ) {
        case 'h':      /* -h (help switch) */
          printUsage();
          break;
        case 'v':      /* -v (verbose switch) */
          parsedArgs->verbose = 1;
          break;
        case 'i':      /* -i <input FITS file> */
          if ( ++i >= argc ) {
            printf("-i <input FITS file> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              sscanf(argv[i], "%s", parsedArgs->inFile );
	  }
          break;
        case 'o':      /* -o <input FITS file> */
          if ( ++i >= argc ) {
            printf("-o <output FITS file> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              sscanf(argv[i], "%s", parsedArgs->outFile );
	  }
          break;
        case 'b':      /* -b <BITPIX> */
          if ( ++i >= argc ) {
            printf("-b <BITPIX> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->bitpix = atoi( argv[i] );
	  }
          break;
        case 'd':      /* -d <data-distribution type> */
          if ( ++i >= argc ) {
            printf("-d <data-distribution type> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->dataOption = atoi( argv[i] );
	  }
          break;
        case 'e':      /* -e <random-number-generator seed> */
          if ( ++i >= argc ) {
            printf("-e <random-number-generator seed> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->iseed = atoi( argv[i] );
	  }
          break;
        case 'n':      /* -n <NAXIS1> */
          if ( ++i >= argc ) {
            printf("-n <NAXIS1> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->naxis1 = atoi( argv[i] );
	  }
          break;
        case 'm':      /* -m <NAXIS2> */
          if ( ++i >= argc ) {
            printf("-m <NAXIS2> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->naxis2 = atoi( argv[i] );
	  }
          break;
        case 'a':      /* -a <mean> */
          if ( ++i >= argc ) {
            printf("-a <mean> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->mu = atof( argv[i] );
	  }
          break;
        case 's':      /* -s <sigma> */
          if ( ++i >= argc ) {
            printf("-s <sigma> missing argument...\n" );
	  } else {
	    if ( argv[i-1][2] == '\0' )
              parsedArgs->sig = atof( argv[i] );
	  }
          break;
        default:
          printf("Unknown argument...\n");
      }
    } else {
      printf( "Command line syntax error:\n" );
      printf( "   Previous argument = %s\n",argv[i - 1] );
      printf( "   Current argument = %s\n",argv[i] );
    }
    i++;
  }

  if (! (strcmp(parsedArgs->outFile,"") != 0 )) {
    printf( "%s %s\n",
	    "*** Error: No output FITS file specified (-o <output FITS file>);",
            "quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if (! ((parsedArgs->bitpix == 16) ||
	 (parsedArgs->bitpix == 8) || 
         (parsedArgs->bitpix == -32))) {
    printf( "%s %s %s\n",
	    "*** Error: BITPIX has invalid value ",
            "(-b <BITPIX>; either 8, 16, or -32 allowed);",
            "quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs->dataOption == 0 ) {
    printf( "%s %s\n",
	    "*** Error: Data-distribution type not specified ",
            "(-d <data-distribution type>); quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs->naxis1 == 0 ) {
    printf( "%s %s\n",
	    "*** Error: NAXIS1 not specified (-n <NAXIS1>);",
            "quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs->naxis1 < 2 ) {
    printf( "%s %s\n",
	    "*** Error: NAXIS1 should be greater than one;",
            "quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs->naxis2 == 0 ) {
    printf( "%s %s\n",
	    "*** Error: NAXIS2 not specified (-m <NAXIS2>);",
            "quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs->naxis2 < 2 ) {
    printf( "%s %s\n",
	    "*** Error: NAXIS2 should be greater than one;",
            "quitting..." );
    exit( TERMINATE_FAILURE );
  }

  if ( parsedArgs->dataOption == 2 ) {
    if ( parsedArgs->mu > 1000 ) {
      printf( "%s %s\n",
	      "*** Error: The table of log factorials must be made",
              "bigger to handle mu > 1000; quitting..." );
      exit( TERMINATE_FAILURE );   
    }
    if ( parsedArgs->mu <= 0.0 ) {
      printf( "%s %s\n",
	      "*** Error: For the Poisson distribution, the",
              "specified mean must be greater than zero; quitting..." );
      exit( TERMINATE_FAILURE );
    }
    parsedArgs->sig = sqrt(parsedArgs->mu);
  }

  printf( "\n%s, v. %s\n\n", PROGRAM, VERSION );
  printf( "Inputs:\n" );
  printf( "   Output FITS file = %s\n", parsedArgs->outFile );
  printf( "   BITPIX = %d\n", parsedArgs->bitpix );
  printf( "   NAXIS1 = %ld\n", parsedArgs->naxis1 );
  printf( "   NAXIS2 = %ld\n", parsedArgs->naxis2 );
  printf( "   Data option = %d\n", parsedArgs->dataOption );
  if ( parsedArgs->dataOption != 3 ) {
     printf( "   Random-number-generator seed = %lu\n", parsedArgs->iseed );
     printf( "   Mean = %f\n", parsedArgs->mu );
     printf( "   Sigma = %f\n", parsedArgs->sig );
  } else {
     printf( "   Input FITS file = %s\n", parsedArgs->inFile );
  }
  printf( "\n" );

}


void printUsage() {
    printf( "\n%s%s%s\n\n", 
            PROGRAM, ", v. ", VERSION );
    printf( "Create a test FITS file with known statistical properties.\n" );
    printf( "The mean and standard deviation are computed for the output\n" );
    printf( "image and written to MEANCOMP and SIGCOMP header keywords,\n" );
    printf( "respectively.\n\n" );
    printf( "%s\n", 
            "Options:" );
    printf( "%s\n", 
            "-i Input FITS file (REQURIED if -d 3 is specified)" );
    printf( "%s\n", 
            "-o Output FITS file (REQURIED)" );
    printf( "%s\n", 
            "-b BITPIX of output FITS file (8, 16, or -32 allowed; default = -32)\n" );
    printf( "-d Data distribution  (REQURIED)\n" );
    printf( "   1 = Randomly sampled from Gaussion distribution\n" );
    printf( "   2 = Randomly sampled from Poisson distribution\n" );
    printf( "   3 = Read from existing FITS file, whose NAXIS# values must\n" );
    printf( "       be same as those specified by the -n and -m options,\n" );
    printf( "       and pass it through to the output FITS file\n" );
    printf( "-e Random-number-generator seed (default = 999999999)\n" );
    printf( "-n NAXIS1 of output FITS file (REQURIED)\n" );
    printf( "-m NAXIS2 of output FITS file (REQURIED)\n" );
    printf( "-a Mean of data distribution (default = 0.0)\n" );
    printf( "-s Standard deviation of data distribution (default = 1.0)\n" );
    printf( "\n" );
    printf( "%s\n", 
            "Switches:" );
    printf( "-h (help)\n" );
    printf( "-v (verbose)\n" );
    printf( "\n" );
    printf( "%s\n", 
            "Notes:" );
    printf( "1. The -s option is ignored if -d 2 is specified, since\n" );
    printf( "   the standard deviation of a Poisson distribution is\n" );
    printf( "   equal to the square root of its mean.\n" );
    printf( "2. The NAXIS# values specified by -n and -m options must be\n" );
    printf( "   equal to those of the input image if -d 3 is specified.\n" );
    printf( "3. The -a and -s options are ignored if -d 3 is specified.\n" );
    printf( "\n" );
    printf( "Required libraries: cfitsio, gsl, nanvalue\n" );
    printf( "\n" );

    exit( TERMINATE_SUCCESS );
}

int nint( double value )
{
  int final;
  if ( value >= 0 ) {
    final = (int) ( value + 0.5 );
  } else {
    final = (int) ( value - 0.5 );
  }
  return final;
}


void computeImageDataStatistics_u8( int npixels,
                                    int *n, 
                                    double *muComp, 
                                    double *sigComp, 
                                    unsigned char *image_u8, 
                                    ParsedArguments *parsedArgs )
{
  int i;
  double diff, sum;

  *muComp = 0.0;
  *n = 0;
  for (i = 0; i < npixels; i++) {
    *muComp += (double) image_u8[i];
    (*n)++;
  }

  if ( *n < 2 ) {
    printf( "%s %s\n",
	    "*** Error: Number of image-data samples should be greater than ",
            "one; quitting..." );
    exit( TERMINATE_FAILURE );
  }

  *muComp /= (double) *n;
  *sigComp = 0.0;
  for (i = 0; i < npixels; i++) {
    diff = (double) image_u8[i] - *muComp;
    *sigComp += diff * diff;
  }
  *sigComp = sqrt( *sigComp / (double) ( *n - 1 ) );
}


void computeImageDataStatistics_u16( int npixels,
                                     int *n, 
                                     double *muComp, 
                                     double *sigComp, 
                                     unsigned short *image_u16, 
                                     ParsedArguments *parsedArgs )
{
  int i;
  double diff, sum;

  *muComp = 0.0;
  *n = 0;
  for (i = 0; i < npixels; i++) {
    *muComp += (double) image_u16[i];
    (*n)++;
  }

  if ( *n < 2 ) {
    printf( "%s %s\n",
	    "*** Error: Number of image-data samples should be greater than ",
            "one; quitting..." );
    exit( TERMINATE_FAILURE );
  }

  *muComp /= (double) *n;
  *sigComp = 0.0;
  for (i = 0; i < npixels; i++) {
    diff = (double) image_u16[i] - *muComp;
    *sigComp += diff * diff;
  }
  *sigComp = sqrt( *sigComp / (double) ( *n - 1 ) );
}


void computeImageDataStatistics_f( int npixels,
                                   int *n, 
                                   double *muComp, 
                                   double *sigComp, 
                                   float *image_f, 
                                   ParsedArguments *parsedArgs )
{
  int i;
  double diff, sum;

  *muComp = 0.0;
  *n = 0;
  for (i = 0; i < npixels; i++) {
    if ( ! ( iznand( image_f[i] ) ) ) {
      *muComp += image_f[i];
      (*n)++;
    }
  }

  if ( *n < 2 ) {
    printf( "%s %s\n",
	    "*** Error: Number of image-data samples should be greater than ",
            "one; quitting..." );
    exit( TERMINATE_FAILURE );
  }

  *muComp /= (double) *n;
  *sigComp = 0.0;
  for (i = 0; i < npixels; i++) {
    if ( ! ( iznand( image_f[i] ) ) ) {
      diff = image_f[i] - *muComp;
    }
    *sigComp += diff * diff;
  }
  *sigComp = sqrt( *sigComp / (double) ( *n - 1 ) );
}


void loadGaussianSamplesIntoImage_u8( int npixels,
                                      double mu,
                                      double sig,
                                      unsigned char *image_u8 ) 
{
  int i, ideviate;
  double deviate;

  for (i = 0; i < npixels; i++) {
    deviate = gsl_ran_gaussian(rng, sig) + mu;
    ideviate = nint( deviate );
    if ( ideviate > 0 ) {
      image_u8[i] = (unsigned char) ideviate;
    } else {
      image_u8[i] = 0;
    }
  }
}


void loadGaussianSamplesIntoImage_u16( int npixels,
                                       double mu,
                                       double sig,
                                       unsigned short *image_u16 ) 
{
  int i, ideviate;
  double deviate;

  for (i = 0; i < npixels; i++) {
    deviate = gsl_ran_gaussian(rng, sig) + mu;
    ideviate = nint( deviate );
    if ( ideviate > 0 ) {
      image_u16[i] = (unsigned short) ideviate;
    } else {
      image_u16[i] = 0;
    }
  }
}


void loadGaussianSamplesIntoImage_f( int npixels,
                                     double mu,
                                     double sig,
                                     float *image_f ) 
{
  int i;
  double deviate;

  for (i = 0; i < npixels; i++) {
    deviate = gsl_ran_gaussian(rng, sig) + mu;
    image_f[i] = (float) deviate;
  }
}


void loadPoissonSamplesIntoImage_u8( int npixels,
                                     double mu,
                                     unsigned char *image_u8 )
{
  int i, ideviate;
  double deviate;

  for (i = 0; i < npixels; i++) {
    deviate = gsl_ran_poisson(rng, mu);
    ideviate = nint( deviate );
    if ( ideviate > 0 ) {
      image_u8[i] = (unsigned char) ideviate;
    } else {
      image_u8[i] = 0;
    }
  }
}


void loadPoissonSamplesIntoImage_u16( int npixels,
                                      double mu,
                                      unsigned short *image_u16 )
{
  int i, ideviate;
  double deviate;

  for (i = 0; i < npixels; i++) {
    deviate = gsl_ran_poisson(rng, mu);
    ideviate = nint( deviate );
    if ( ideviate > 0 ) {
      image_u16[i] = (unsigned short) ideviate;
    } else {
      image_u16[i] = 0;
    }
  }
}


void loadPoissonSamplesIntoImage_f( int npixels,
                                    double mu,
                                    float *image_f )
{
  int i;
  double deviate;

  for (i = 0; i < npixels; i++) {
    deviate = gsl_ran_poisson(rng, mu);
    image_f[i] = (float) deviate;
  }
}


