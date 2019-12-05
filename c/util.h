#ifndef __util_H__
#define __util_H__

void add_arrays(const int size, const double *arr1, double *arr2);
void multiply_array(const int size, const double scalar, double *arr);
void copy_array(const int size, const double *arr, double *copy);
double det_array(const double arr[9]);
int is_gamma(const double qpt[3]);
double cell_volume(const double cell_vec[9]);
void cmult(const double c1[2], const double c2[2], double *result);
void cmult_conj(const double c1[2], const double c2[2], double *result);

#endif
