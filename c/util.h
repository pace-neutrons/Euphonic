#ifndef __util_H__
#define __util_H__

void add_arrays(const int size, const double *arr1, double *arr2);
void multiply_array(const int size, const double scalar, double *arr);
double det_array(const double arr[9]);
int is_gamma(const double *qpt);

#endif
