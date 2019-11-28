#include <math.h>

void add_arrays(const int size, const double *arr1, double *arr2) {
    int i;
    for (i = 0; i < size; i++) {
        arr2[i] += arr1[i];
    }
}

void multiply_array(const int size, const double scalar, double *arr) {
    int i;
    for (i = 0; i < size; i++) {
        arr[i] *= scalar;
    }
}

double det_array(const double arr[9]) {
   double det;
   det = arr[0]*(arr[4]*arr[8] - arr[7]*arr[5]) +
         arr[1]*(arr[3]*arr[8] - arr[6]*arr[5]) +
         arr[2]*(arr[3]*arr[7] - arr[6]*arr[4]);
   return det;
}

int is_gamma(const double *qpt) {
     const double tol = 1e-15;
     int i;
     double diff = 0;

     for (i = 0; i < 3; i++) {
         diff += fabs(qpt[i] - round(qpt[i]));
     }

     if (diff < tol) {
         return 1;
     }
     return 0;
}
