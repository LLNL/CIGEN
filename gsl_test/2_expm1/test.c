
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gsl/gsl_sys.h"



// modify parameter here
void compute(double var_1) {
  // modify function + parameter here
  double comp = gsl_expm1(var_1);
  #ifdef TEXT_OUTPUT
  printf("%.17g\n", comp);
  #else
  printf("%#08llX\n", *(long long*)&comp);
  #endif
}

int main(int argc, char** argv) {
/* Program variables */

  long long tmp_1_i = strtoull(argv[1], NULL, 16); double tmp_1 = *(double*)&tmp_1_i;

  compute
  (tmp_1);


  return 0;
}
