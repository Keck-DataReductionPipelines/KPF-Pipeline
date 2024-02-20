#include <string.h>
#include <stdio.h>
#include <strings.h>
#include "fitsio.h"

#define MAX_LINE_LENGTH 1024

int main(int argc, char *argv[])
{
    fitsfile *fptr;         /* FITS file pointer, defined in fitsio.h */
    char card[FLEN_CARD], newcard[FLEN_CARD], keyword[9];
    char newvalue[FLEN_VALUE], oldvalue[FLEN_VALUE], comment[FLEN_COMMENT];
    char line[MAX_LINE_LENGTH];
    char *line_ptr, *p;
    int status = 0;   /*  CFITSIO status value MUST be initialized to zero!  */
    int iomode, keytype;
    FILE *fp;

    iomode = READWRITE;

    if (argc < 3)
    {
      printf("Usage:  hdrupdate filename -a|-d txtfile (or -c switch to only update checksums)\n");
      printf("\n");
      return(0);
    }

    /* open_file */
    if (fits_open_file(&fptr, argv[1], iomode, &status))
    {
       /* if error occured, print out error message */
       if (status) fits_report_error(stderr, status);
       exit(64);
    }

    if (argc == 4)
    {
        if((fp = fopen(argv[3], "r"))==NULL) {
          perror("hdrupdate");
          exit(64);
        }

        while(1)
        {
          line_ptr = fgets(line, sizeof(line), fp);
          if((*line=='\0')||(line_ptr==NULL))
            break;

          if (line[strlen(line)-1] == '\n') line[strlen(line)-1] = '\0';
          *keyword = '\0';
          strncpy(keyword, line, 8);
          keyword[8] = '\0';
          strcpy(newvalue, line+10);

          /* if -d, delete the keyword */

          if (argv[2][1] == 'd') {
            printf("Deleting card with keyword=[%s]...\n", keyword);
	    fits_delete_key(fptr,keyword, &status);
            if (status)fits_report_error(stderr, status);
            if (status == 202) {
              status = 0;  /* reset status after error */
	    }
          }

          else if (argv[2][1] == 'a') {
             if (fits_read_card(fptr, keyword, card, &status))
             {
               card[0] = '\0';
               comment[0] = '\0';
               status = 0;  /* reset status after error */
             }

             /* check if this is a protected keyword that must not be changed */
             if (*card && fits_get_keyclass(card) == TYP_STRUC_KEY)
             {
                printf("Protected keyword cannot be modified.\n");
                exit(64);
             }
             else
             {
                /* get the comment string */
                if (*card)fits_parse_value(card, oldvalue, comment, &status);

                /* construct template for new keyword */
                strcpy(newcard, keyword);      /* copy keyword name */
                strcat(newcard, " = ");        /* '=' value delimiter */
                strcat(newcard, newvalue);     /* new value */
                if (*comment) {	
                  p = strchr(newcard, '/');
                  if (p!=NULL)
	          *(p++)='\0';		
                  strcat(newcard, " / ");
                  strcat(newcard, comment);    /* append the old comment */
                }

                /* reformat the keyword string to conform to FITS rules */
                fits_parse_template(newcard, card, &keytype, &status);

                /* overwrite the keyword with the new value */
                printf("Adding card=[%s]...\n", card);
                fits_update_card(fptr, keyword, card, &status);
             }
           }
        }
    }

    printf("Updating FITS checksums...\n");
    fits_update_chksum(fptr, &status);
    if (status == 202) {
        status = 0;  /* reset status after error */
        fits_write_chksum(fptr, &status);
    }

    fclose(fp);
    fits_close_file(fptr, &status);
    return(status);
}

