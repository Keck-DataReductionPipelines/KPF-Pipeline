/* IMPORTANT CAAVEAT:
   This library must be linked as a system library after all others
*/





/* mask for exponent in an IEEE float */
const union {unsigned long i; float f;  } const_exp_float =  {0x7f800000};
/* nan IEEE float */
const union {unsigned long i; float f;  } const_nan_float =  {0x7fc00000};
/* IEEE float mantissa */
const union {unsigned long i; float f;  } const_mant_float = {0x007fffff};
/* mask for exponent in an IEEE double */
const union {unsigned long long i; double f;  } const_exp_double =  {0x7ff0000000000000LL};
/* nan IEEE double */
const union {unsigned long long i; double f;  } const_nan_double =  {0x7ff8000000000000LL};
/* IEEE double infinity */
const union {unsigned long long i; double f;  } const_mant_double = {0x000fffffffffffffLL};

int iznanf(float f) {
  union {unsigned long i; float f;  } u;
  unsigned long exp,mant;
  u.f=f;
  exp=u.i&const_exp_float.i;
  mant=u.i&const_mant_float.i;
  return (exp==const_exp_float.i) && (mant!=0);
}
/* iznan functions iznanf and iznand
   test for exponent to be the maximum value and nonzero mantissa - does not count infinity as nan 
*/
int iznand(double f) {
  union {unsigned long long i; double f;  } u;
  unsigned long long exp,mant;
  u.f=f;
  exp=u.i&const_exp_double.i;
  mant=u.i&const_mant_double.i;
  return (exp==const_exp_double.i) && (mant!=0);
}

/* izinf function izinff and izinfd
   test for exponent being max and zero mantissa; does not count nan as infinity
*/
int izinff(float f) {
  union {unsigned long i; float f;  } u;
  unsigned long exp,mant;
  u.f=f;
  exp=u.i&const_exp_float.i;
  mant=u.i&const_mant_float.i;
  return (exp==const_exp_float.i) && (mant==0) ;
}

int izinfd(double f) {
  union {unsigned long long i; double f;  } u;
  unsigned long long exp,mant;
  u.f=f;
  exp=u.i&const_exp_double.i;
  mant=u.i&const_mant_double.i;
  return (exp==const_exp_double.i) && (mant==0);
}

/* returns 1 if the input is a nan or an infinity */
int iznanorinff(float f){
  union {unsigned long i; float f;  } u;
  unsigned long exp,mant;
  u.f=f;
  exp=u.i&const_exp_float.i;
  mant=u.i&const_mant_float.i;
  return (exp==const_exp_float.i) ;
}
int iznanorinfd(double d){
  union {unsigned long long i; double f;  } u;
  unsigned long long exp,mant;
  u.f=d;
  exp=u.i&const_exp_double.i;
  mant=u.i&const_mant_double.i;
  return (exp==const_exp_double.i);
}
