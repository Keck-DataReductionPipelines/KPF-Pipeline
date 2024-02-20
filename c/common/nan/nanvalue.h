#ifndef __NANVALUE
#define __NANVALUE

/* a way to define a NAN for all machines with IEEE floating point 
   works on both big-endian and little-endian variants because the 
   byte order of float is the same as the byte order of long
   
   The actual constants are defined in the sourcefile nanvalue.c
*/

#ifdef	__cplusplus
extern "C" {
#endif

/* mask for exponent in an IEEE float */
extern const union {unsigned long i; float f;  } const_exp_float;
/* nan IEEE float */
extern const union {unsigned long i; float f;  } const_nan_float ;
/* IEEE float mantissa */
extern const union {unsigned long i; float f;  } const_mant_float;
/* mask for exponent in an IEEE double */
extern const union {unsigned long long i; double f;  } const_exp_double;
/* nan IEEE double */
extern const union {unsigned long long i; double f;  } const_nan_double;
/* IEEE double infinity */
extern const union {unsigned long long i; double f;  } const_mant_double;


#undef NANVALUE
#define NANVALUE (const_nan_float.f)

#undef INFVALUE
#define INFVALUE (const_mant_float.f)


/* Needed for OS X. To quote Bill Northcott:

isnan() is a C99 extension to standard C.
Standard C++ is based on an older standard of C.
Hence isnan() is not part of standard C++ and may or may not work.

The C++ equivalent of math.h is cmath. cmath on MacOS X and probably other platforms undefines isnan() and other C99 macros. Hence if your program includes cmath or one of the many headers such iostream that themselves include cmath then isnan() will not be available.

The code works on Linux because isnan() is a function not a macro and 
the macro expansion takes place in the preprocessor before the compiler sees it.

The solution for any math library for C++ (such as libRmath) is to program your own isnan() like function that access the underlying system functions.
*/

/* iznan functions in its float and double variations: iznanf and iznand
   test for exponent to be the maximum value and nonzero mantissa - does not count infinity as nan 
*/
int iznanf(float f) ;
int iznand(double f) ;

/* izinf function  in its float and double variations: izinff and izinfd
   test for exponent being max and zero mantissa; does not count nan as infinity
*/
int izinff(float f) ;
int izinfd(double f) ;

/* returns 1 if the input is a nan or an infinity */
int iznanorinff(float f);
int iznanorinfd(double d);

/* signum function in two variations: signanorinff and signanorinfd
   returns +1 if the sign bit is zero which corresponds to a positive number, 
           -1 if the sign bit is set  which corresponds to a negative number
*/
int signanorinff(float f);
int signanorinfd(double d);

/* allow a way with a command-line preprocessor symbol to define the isnan function
   in terms of the iznan series of library routines referred to above
*/

#ifdef _DEFINE_ISNAN
#undef isnan
#define isnan(X) iznanf(X)
#endif


#ifdef __cplusplus
}
#endif


#endif
