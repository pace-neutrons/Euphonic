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

void copy_array(const int size, const double *arr, double *copy) {
    int i;
    for (i = 0; i < size; i++) {
        copy[i] = arr[i];
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

double cell_volume(const double *cell_vec) {
    // Assume cell_vec is in order ax, ay, az, bz, by...
    int i;
    double bxc[3];
    double vol = 0;

    bxc[0] = cell_vec[4]*cell_vec[8] - cell_vec[5]*cell_vec[7];
    bxc[1] = cell_vec[5]*cell_vec[6] - cell_vec[3]*cell_vec[8];
    bxc[2] = cell_vec[3]*cell_vec[7] - cell_vec[4]*cell_vec[6];

    for (i = 0; i < 3; i++) {
        vol += cell_vec[i]*bxc[i];
    }
    return vol;
}

void cmult(const double c1[2], const double c2[2], double result[2]) {
    result[0] = c1[0]*c2[0] - c1[1]*c2[1];
    result[1] = c1[0]*c2[1] + c1[1]*c2[0];
}

void cmult_conj(const double c1[2], const double c2[2], double result[2]) {
    result[0] =  c1[0]*c2[0] + c1[1]*c2[1];
    result[1] = -c1[0]*c2[1] + c1[1]*c2[0];
}
