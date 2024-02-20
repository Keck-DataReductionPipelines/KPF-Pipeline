/* $Id: verifyHduSums.h,v 1.1 2016/05/10 17:46:37 laher Exp $ */

#include "fitsio.h"


/* Constants. */

#define CODENAME "verifyHduSums"
#define VERSION  1.0

#define TERMINATE_SUCCESS  0
#define TERMINATE_WARNING  32
#define TERMINATE_FAILURE  64


/* Prototypes. */

int printUsage( char codeName[], char version[] );


