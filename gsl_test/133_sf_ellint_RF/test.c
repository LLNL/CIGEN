
/* This is a automatically generated test. Do not modify */


#define COMPILE_INLINE_STATIC

#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#include "gsl/gsl_sf_ellint.h"



// modify parameter here
void compute(double var_1, double var_2, double var_3) {
  // modify function + parameter here
  double comp = gsl_sf_ellint_RF(var_1, var_2, var_3, GSL_PREC_DOUBLE);
  #ifdef TEXT_OUTPUT
  printf("%.17g\n", comp);
  #else
  printf("%#08llX\n", *(long long*)&comp);
  #endif
}

int main(int argc, char** argv) {
/* Program variables */

  long long tmp_1_i = strtoull(argv[1], NULL, 16); double tmp_1 = *(double*)&tmp_1_i;
  long long tmp_2_i = strtoull(argv[2], NULL, 16); double tmp_2 = *(double*)&tmp_2_i;
  long long tmp_3_i = strtoull(argv[3], NULL, 16); double tmp_3 = *(double*)&tmp_3_i;

  compute
  (tmp_1, tmp_2, tmp_3);


  return 0;
}
