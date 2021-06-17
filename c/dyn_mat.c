#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "load_libs.h"

#define PI 3.14159265358979323846

void calculate_dyn_mat_at_q(const double *qpt, const int n_atoms,
    const int n_cells, const int max_images, const int *n_sc_images,
    const int *sc_image_i, const int *cell_origins, const int *sc_origins,
    const double *fc_mat, const double *all_origins_cart, const bool calc_dmat_grad,
    double *dyn_mat, double *dmat_grad) {

    int i, j, n, nc, k, sc, ii, jj, sc_img_idx, idx, idx_t;
    double qdotr;
    double rcart;
    double phase[2];
    double phase_sum[2];
    double rcart_sum[6];

    // Note: C calculated dynamical matrix uses e^-i(q.r) convention, whereas
    // Python uses the e^i(q.r) convention. This differing convention is used
    // because Scipy defines the Fortran interface to zheevd so expects Fortran
    // ordering (dyn mat is Hermitian so transpose = complex conjugate)

    // Array strides
    int s_n[2] = {n_atoms*n_atoms, n_atoms}; // For n_sc_images
    // For sc_image_i
    int s_i[3] = {n_atoms*n_atoms*max_images, n_atoms*max_images, max_images};
    int s_fc = 9*n_atoms*n_atoms; // For fc_mat

    memset(dyn_mat, 0, 2*9*n_atoms*n_atoms*sizeof(double));
    if (calc_dmat_grad) {
        memset(dmat_grad, 0, 3*2*9*n_atoms*n_atoms*sizeof(double));
    }
    for (i = 0; i < n_atoms; i++) {
        for (j = i; j < n_atoms; j++) {
            for (nc = 0; nc < n_cells; nc++){
                memset(phase_sum, 0, 2*sizeof(double));
                memset(rcart_sum, 0, 6*sizeof(double));
                // Calculate and sum phases for all  images
                for (n = 0; n < n_sc_images[nc*s_n[0] + i*s_n[1] + j]; n++) {
                    qdotr = 0;
                    sc_img_idx = nc*s_i[0] + i*s_i[1] + j*s_i[2] + n;
                    sc = sc_image_i[sc_img_idx];
                    for (k = 0; k < 3; k++){
                        qdotr += qpt[k]*(sc_origins[3*sc + k] + cell_origins[3*nc + k]);
                    }
                    phase[0] = cos(2*PI*qdotr);
                    phase[1] = -sin(2*PI*qdotr);
                    phase_sum[0] += phase[0];
                    phase_sum[1] += phase[1];
                    if (calc_dmat_grad) {
                        for (k = 0; k < 3; k++){
                            // Note: use cos + isin phase as dyn mat gradients aren't passed
                            // to a Fortran lib so we need to use the e^i(q.r) convention
                            rcart = all_origins_cart[3*sc_img_idx + k];
                            rcart_sum[2*k] += phase[1]*rcart; //Multiply phase by i: swap re and im
                            rcart_sum[2*k + 1] += phase[0]*rcart;
                        }
                    }
                }
                for (ii = 0; ii < 3; ii++){
                    for (jj = 0; jj < 3; jj++){
                        idx = (3*i+ii)*3*n_atoms + 3*j + jj;
                        // Real part
                        dyn_mat[2*idx] += phase_sum[0]*fc_mat[nc*s_fc + idx];
                        // Imaginary part
                        dyn_mat[2*idx + 1] += phase_sum[1]*fc_mat[nc*s_fc + idx];
                        if (calc_dmat_grad) {
                            for (k = 0; k < 3; k++) {
                                // Real
                                dmat_grad[6*idx + 2*k] += rcart_sum[2*k]*fc_mat[nc*s_fc + idx];
                                // Imaginary
                                dmat_grad[6*idx + 2*k + 1] += rcart_sum[2*k + 1]*fc_mat[nc*s_fc + idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Fill in lower triangular of dmat gradients - is Hermitian
    if (calc_dmat_grad) {
        for (i = 1; i < n_atoms; i++) {
            for (j = 0; j < i; j++) {
                for (ii = 0; ii < 3; ii++){
                    for (jj = 0; jj < 3; jj++){
                        idx = 6*((3*i+ii)*3*n_atoms + 3*j + jj);
                        idx_t = 6*((3*j+jj)*3*n_atoms + 3*i + ii);
                        for (k = 0; k < 3; k++) {
                            dmat_grad[idx + 2*k] = dmat_grad[idx_t + 2*k];
                            dmat_grad[idx + 2*k + 1] = -dmat_grad[idx_t + 2*k + 1];
                        }
                    }
                }
            }
        }
    }
}

void calculate_dipole_correction(const double *qpt, const int n_atoms,
    const double *cell_vec, const double *recip, const double *atom_r,
    const double *born, const double *dielectric, const double *H_ab,
    const double *cells, const int n_dcells, const double *gvec_phases,
    const double *gvecs_cart, const int n_gvecs, const double *dipole_q0,
    const double lambda, double *corr) {

    int size = 2*9*n_atoms*n_atoms;
    double q_cart[3] = {0, 0, 0};
    double qpt_norm[3];
    double qdotr;
    double phase[2];
    const double *H_ab_ptr;
    double det_e;
    int n_gvecs_local;
    int i, j, a, b, aa, bb, nc, ng, idx;

    // Normalise q-point
    for (a = 0; a < 3; a++) {
        qpt_norm[a] = qpt[a] - round(qpt[a]);
    }

    // Don't include G=0 vector if q=0
    n_gvecs_local = n_gvecs;
    if (is_gamma(qpt_norm)) {
        gvec_phases += 2*n_atoms;
        gvecs_cart += 3;
        n_gvecs_local--;
    }

    // Calculate realspace term
    memset(corr, 0, 2*9*n_atoms*n_atoms*sizeof(double));
    H_ab_ptr = H_ab;
    for (nc = 0; nc < n_dcells; nc++) {
        qdotr = 0;
        for (a = 0; a < 3; a++) {
            qdotr += qpt_norm[a]*cells[3*nc + a];
        }
        phase[0] = cos(2*PI*qdotr);
        phase[1] = -sin(2*PI*qdotr);

        for (i = 0; i < n_atoms; i++) {
            for (j = i; j < n_atoms; j++) {
                for (a = 0; a < 3; a++) {
                    for (b = 0; b < 3; b++) {
                        idx = 2*(3*(3*i + a)*n_atoms + 3*j + b);
                        corr[idx] -= phase[0]*(*H_ab_ptr);
                        corr[idx + 1] -= phase[1]*(*H_ab_ptr);
                        H_ab_ptr++;
                    }
                }
            }
        }
    }
    det_e = det_array(dielectric);
    multiply_array(size, pow(lambda, 3)/sqrt(det_e), corr);

    // Precalculate some values for reciprocal space term
    // Calculate dielectric/4(lambda^2) factor
    double diel_lambda[9];
    for (a = 0; a < 9; a++) {
        diel_lambda[a] = dielectric[a]/(4*pow(lambda, 2));
    }
    // Calculate q in Cartesian coords
    for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) {
            q_cart[a] += qpt_norm[b]*recip[3*b + a];
        }
    }
    // Calculate q-point phases
    double *q_phases;
    q_phases = (double*)malloc(2*n_atoms*sizeof(double));
    for (i = 0; i < n_atoms; i++) {
        qdotr = 0;
        for (a = 0; a < 3; a++) {
            qdotr += qpt_norm[a]*atom_r[3*i + a];
        }
        q_phases[2*i] = cos(2*PI*qdotr);
        q_phases[2*i + 1] = -sin(2*PI*qdotr);
    }
    // Calculate reciprocal term multiplication factor
    double fac = PI/(cell_volume(cell_vec)*pow(lambda, 2));
    // Calculate reciprocal term
    double kvec[3];
    double k_ab_exp[9];
    double k_len_2;
    double gq_phase_ri[2];
    double gq_phase_rj[2];
    double gq_phase_rij[2];
    for (ng = 0; ng < n_gvecs_local; ng++) {

        for (a = 0; a < 3; a++) {
            kvec[a] = gvecs_cart[3*ng + a] + q_cart[a];
        }
        k_len_2 = 0;
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++) {
                idx= 3*a + b;
                k_ab_exp[idx] = kvec[a]*kvec[b];
                k_len_2 += k_ab_exp[idx]*diel_lambda[idx];
            }
        }
        multiply_array(9, exp(-k_len_2)/k_len_2, k_ab_exp);

        for (i = 0; i < n_atoms; i++) {
            idx = 2*(ng*n_atoms + i);
            // Due to differing phase conventions in Python/C, as gvec_phases
            // was precalculated in Python, must use the complex conj
            cmult_conj((q_phases + 2*i), (gvec_phases + idx), gq_phase_ri);
            for (j = i; j < n_atoms; j++) {
                idx = 2*(ng*n_atoms + j);
                cmult_conj((q_phases + 2*j), (gvec_phases + idx), gq_phase_rj);
                // To divide by gq_phase_rj, multiply by complex conj
                cmult_conj(gq_phase_ri, gq_phase_rj, gq_phase_rij);
                for (a = 0; a < 3; a++) {
                    for (b = 0; b < 3; b++) {
                        idx = 2*(3*(3*i + a)*n_atoms + 3*j + b);
                        corr[idx] += fac*gq_phase_rij[0]*k_ab_exp[3*a + b];
                        corr[idx + 1] += fac*gq_phase_rij[1]*k_ab_exp[3*a + b];
                    }
                }
            }
        }
    }
    free((void*)q_phases);

    // Multiply in born charges
    double corr_tmp[18];
    double born_fac;
    for (i = 0; i < n_atoms; i++) {
        for (j = i; j < n_atoms; j++) {
            memset(corr_tmp, 0, 18*sizeof(double));
            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    for (aa = 0; aa < 3; aa++) {
                        for (bb = 0; bb < 3; bb++) {
                            idx = 2*(3*(3*i + aa)*n_atoms + 3*j + bb);
                            born_fac = born[9*i + 3*a + aa]*born[9*j + 3*b + bb];
                            corr_tmp[6*a + 2*b] += born_fac*corr[idx];
                            corr_tmp[6*a + 2*b + 1] += born_fac*corr[idx + 1];
                        }
                    }
                }
            }

            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    idx = 2*(3*(3*i + a)*n_atoms + 3*j + b);
                    corr[idx] = corr_tmp[6*a + 2*b];
                    corr[idx + 1] = corr_tmp[6*a + 2*b + 1];
                }
            }

        }

        // Subtract q=0 correction from diagonal
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++) {
                idx = 2*(3*(3*i + a)*n_atoms + 3*i + b);
                corr[idx] -= dipole_q0[18*i + 6*a + 2*b];
                corr[idx + 1] -= dipole_q0[18*i + 6*a + 2*b + 1];
            }
        }
    }

}

void calculate_gamma_correction(const double q_dir[3], const int n_atoms,
    const double *cell_vec, const double *recip_vec, const double *born,
    const double *dielectric, double *corr) {

    int i, j, a, b, idx;
    double fac, denominator;
    double *q_born_sum;
    double q_dir_cart[3];

    if (is_gamma(q_dir)) {
        memset(corr, 0, 2*9*n_atoms*n_atoms*sizeof(double));
        return;
    }

    for (a = 0; a < 3; a++) {
        q_dir_cart[a] = 0;
        for (b = 0; b < 3; b++) {
            q_dir_cart[a] += recip_vec[3*b + a]*q_dir[b];
        }
    }

    denominator = 0;
    for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) {
           denominator += dielectric[3*a + b]*q_dir_cart[a]*q_dir_cart[b];
        }
    }
    fac = (4*PI)/(cell_volume(cell_vec)*denominator);

    q_born_sum = (double*)calloc(3*n_atoms, sizeof(double));
    memset(q_born_sum, 0, 3*n_atoms*sizeof(double));
    for (i = 0; i < n_atoms; i++) {
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++) {
                q_born_sum[3*i + a] += born[9*i + 3*a + b]*q_dir_cart[b];
            }
        }
    }

    for (i = 0; i < n_atoms; i++) {
        for (j = i; j < n_atoms; j++) {
            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    idx = 2*(3*(3*i + a)*n_atoms + 3*j + b);
                    corr[idx] = fac*q_born_sum[3*i + a]*q_born_sum[3*j + b];
                }
            }
        }
    }
    free((void*)q_born_sum);
}

void mass_weight_dyn_mat(const double* dyn_mat_weighting, const int n_atoms,
    const int repeats, double* dyn_mat) {

    int i, j;
    for (i = 0; i < 9*n_atoms*n_atoms; i++) {
        for (j = 0; j < repeats; j++) {
            // Repeats: how many elements of dyn_mat per dyn_mat_weighting
            // As dyn_mat = complex and weighting = real, this is usually 2
            dyn_mat[repeats*i + j] *= dyn_mat_weighting[i];
        }
    }
}

int diagonalise_dyn_mat_zheevd(const int n_atoms, const double qpt[3],
    double* dyn_mat, double* eigenvalues, ZheevdFunc zheevdptr) {

    char jobz = 'V';
    char uplo = 'L';
    int order = 3*n_atoms;
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
        printf("INFO: Zheevd failed querying workspace with info %i at "
               "q-point %f %f %f\n", info, qpt[0], qpt[1], qpt[2]);
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

    if (info != 0) {
       printf("INFO: Zheevd diagonalisation failed with info %i at "
              "q-point %f %f %f\n", info, qpt[0], qpt[1], qpt[2]);
    }

    return info;
}

void evals_to_freqs(const int n_atoms, double *eigenvalues) {
    int i;
    double tmp;
    for (i = 0; i < 3*n_atoms; i++) {
        // Set imaginary frequencies to negative
        tmp = copysign(sqrt(fabs(eigenvalues[i])), eigenvalues[i]);
        eigenvalues[i] = tmp;
    }
}

void calculate_mode_gradients(const int n_atoms, const double *evals,
    const double *evecs, const double *dmat_grad, double *modeg) {
    int n, i, j, a, b, k, grad_idx;
    int n_modes = 3*n_atoms;
    double evec_mult_tmp[2];
    double conj_tmp[2];
    int mode_s = 2*3*n_atoms; //Eigenvector array stride

    for (n = 0; n < n_modes; n++) {
        for (i = 0; i < n_atoms; i++) {
            for (a = 0; a < 3; a++) {
                for (j = 0; j < n_atoms; j++) {
                    for (b = 0; b < 3; b++) {

                        for (k = 0; k < 3; k++) {
                            cmult_conj((evecs + (n*mode_s + 6*j + 2*b)),
                                       (evecs + (n*mode_s + 6*i + 2*a)),
                                       evec_mult_tmp);
                            grad_idx = 3*(3*i + a)*mode_s + 3*(6*j + 2*b) + 2*k;
                            cmult((dmat_grad + grad_idx), evec_mult_tmp, conj_tmp);
                            modeg[6*n + 2*k] += 0.5*conj_tmp[0]/evals[n];
                            modeg[6*n + 2*k + 1] += 0.5*conj_tmp[1]/evals[n];
                        }
                    }
                }
            }
        }
    }
}

