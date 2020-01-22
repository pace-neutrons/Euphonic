#ifndef __load_libs_H__
#define __load_libs_H__

typedef void (*ZheevdFunc)(char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, double* rwork, int* lrwork,
    int* iwork, int* liwork, int* info);

ZheevdFunc get_zheevd(const char *scipy_dir);

#endif
