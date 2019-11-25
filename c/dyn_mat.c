#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

void calculate_dyn_mat_at_q(const double *qpt, const int n_ions,
    const int n_cells, const const int max_images, const int *n_sc_images,
    const int *sc_image_i, const int *cell_origins, const int *sc_origins,
    const double *fc_mat, double *dyn_mat) {

    int i, j, n, nc, k, sc, ii, jj, idx;
    double qdotr;
    double phase_r;
    double phase_i;

    // Array strides
    int s_n[2] = {n_ions*n_ions, n_ions}; // For n_sc_images
    // For sc_image_i
    int s_i[3] = {n_ions*n_ions*max_images, n_ions*max_images, max_images};
    int s_fc = 9*n_ions*n_ions; // For fc_mat

    for (i = 0; i < n_ions; i++) {
        for (j = i; j < n_ions; j++) {
            for (nc = 0; nc < n_cells; nc++){
                phase_r = 0;
                phase_i = 0;
                // Calculate and sum phases for all  images
                for (n = 0; n < n_sc_images[nc*s_n[0] + i*s_n[1] + j]; n++) {
                    qdotr = 0;
                    sc = sc_image_i[nc*s_i[0] + i*s_i[1] + j*s_i[2] + n];
                    for (k = 0; k < 3; k++){
                        qdotr += qpt[k]*(sc_origins[3*sc + k] + cell_origins[3*nc + k]);
                    }
                    phase_r += cos(2*PI*qdotr);
                    phase_i -= sin(2*PI*qdotr);
                }
                for (ii = 0; ii < 3; ii++){
                    for (jj = 0; jj < 3; jj++){
                        idx = (3*i+ii)*3*n_ions + 3*j + jj;
                        // Real part
                        dyn_mat[2*idx] += phase_r*fc_mat[nc*s_fc + idx];
                        // Imaginary part
                        dyn_mat[2*idx + 1] += phase_i*fc_mat[nc*s_fc + idx];
                    } // jj
                } // ii
            } // nc
        } // j
    } // i
}

void mass_weight_dyn_mat(const double* dyn_mat_weighting, const int n_ions,
    double* dyn_mat) {
        int i, j;
        for (i = 0; i < 9*n_ions*n_ions; i++) {
            for (j = 0; j < 2; j++) {
                dyn_mat[2*i + j] *= dyn_mat_weighting[i];
            }
        }
    }

int diagonalise_dyn_mat_zheevd(const int n_ions, double* dyn_mat,
    double* eigenvalues,
    void (*zheevdptr) (char*, char*, int*, double*, int*, double*, double*,
    int*, double*, int*, int*, int*, int*)) {

    char jobz = 'V';
    char uplo = 'L';
    int order = 3*n_ions;
    int lda = order;
    int lwork, lrwork, liwork = -1;
    double *work, *rwork;
    int *iwork;
    int info;

    // Query vars
    double lworkopt, lrworkopt;
    int liworkopt;

    // Workspace query
    (*zheevdptr)(&jobz, &uplo, &order, dyn_mat, &lda, eigenvalues, &lworkopt, &lwork,
        &lrworkopt, &lrwork, &liworkopt, &liwork, &info);
    if (info != 0) {
        printf("Failed querying workspace\n");
        return info;
    }
    lwork = (int)lworkopt;
    lrwork = (int)lrworkopt;
    liwork = liworkopt;

    // Allocate work arrays
    work = (double*)malloc(2*lwork*sizeof(double));
    rwork = (double*)malloc(lrwork*sizeof(double));
    iwork = (int*)malloc(liwork*sizeof(int));

    (*zheevdptr)(&jobz, &uplo, &order, dyn_mat, &lda, eigenvalues, work, &lwork,
        rwork, &lrwork, iwork, &liwork, &info);

    free((void*)work);
    free((void*)rwork);
    free((void*)iwork);

    return info;
}

void evals_to_freqs(const int n_ions, double *eigenvalues) {
    int i;
    double tmp;
    for (i = 0; i < 3*n_ions; i++) {
        // Set imaginary frequencies to negative
        tmp = copysign(sqrt(fabs(eigenvalues[i])), eigenvalues[i]);
        eigenvalues[i] = tmp;
    }
}