#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "load_libs.h"

#define PI 3.14159265358979323846

void calculate_dyn_mat_at_q(const double *qpt, const int n_ions,
    const int n_cells, const const int max_images, const int *n_sc_images,
    const int *sc_image_i, const int *cell_origins, const int *sc_origins,
    const double *fc_mat, double *dyn_mat) {

    int i, j, n, nc, k, sc, ii, jj, idx;
    double qdotr;
    double phase_r;
    double phase_i;

    // Note: C calculated dynamical matrix uses e^-i(q.r) convention, whereas
    // Python uses the e^i(q.r) convention. This differing convention is used
    // because Scipy defines the Fortran interface to zheevd so expects Fortran
    // ordering (dyn mat is Hermitian so transpose = complex conjugate)

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
                    }
                }
            }
        }
    }
}

void calculate_dipole_correction(const double *qpt, const int n_ions,
    const double *cell_vec, const double *recip, const double *ion_r,
    const double *born, const double *dielectric, const double *H_ab,
    const double *cells, const int n_dcells, const double *gvec_phases,
    const double *gvecs_cart, const int n_gvecs, const double *dipole_q0,
    const double eta, double *corr) {

    int size = 2*9*n_ions*n_ions;
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
        gvec_phases += 18;
        gvecs_cart += 3;
        n_gvecs_local--;
    }

    // Calculate realspace term
    memset(corr, 0, 2*9*n_ions*n_ions*sizeof(double));
    H_ab_ptr = H_ab;
    for (nc = 0; nc < n_dcells; nc++) {
        qdotr = 0;
        for (a = 0; a < 3; a++) {
            qdotr += qpt_norm[a]*cells[3*nc + a];
        }
        phase[0] = cos(2*PI*qdotr);
        phase[1] = -sin(2*PI*qdotr);

        for (i = 0; i < n_ions; i++) {
            for (j = i; j < n_ions; j++) {
                for (a = 0; a < 3; a++) {
                    for (b = 0; b < 3; b++) {
                        idx = 2*(3*(3*i + a)*n_ions + 3*j + b);
                        corr[idx] -= phase[0]*(*H_ab_ptr);
                        corr[idx + 1] -= phase[1]*(*H_ab_ptr);
                        H_ab_ptr++;
                    }
                }
            }
        }
    }
    det_e = det_array(dielectric);
    multiply_array(size, pow(eta, 3)/sqrt(det_e), corr);

    // Precalculate some values for reciprocal space term
    // Calculate dielectric/4(eta^2) factor
    double diel_eta[9];
    for (a = 0; a < 9; a++) {
        diel_eta[a] = dielectric[a]/(4*pow(eta, 2));
    }
    // Calculate q in Cartesian coords
    for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) {
            q_cart[a] += qpt_norm[b]*recip[3*b + a];
        }
    }
    // Calculate q-point phases
    double *q_phases;
    q_phases = (double*)malloc(2*n_ions*sizeof(double));
    for (i = 0; i < n_ions; i++) {
        qdotr = 0;
        for (a = 0; a < 3; a++) {
            qdotr += qpt_norm[a]*ion_r[3*i + a];
        }
        q_phases[2*i] = cos(2*PI*qdotr);
        q_phases[2*i + 1] = -sin(2*PI*qdotr);
    }
    // Calculate reciprocal term multiplication factor
    double fac = PI/(cell_volume(cell_vec)*pow(eta, 2));
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
                k_len_2 += k_ab_exp[idx]*diel_eta[idx];
            }
        }
        multiply_array(9, exp(-k_len_2)/k_len_2, k_ab_exp);

        for (i = 0; i < n_ions; i++) {
            idx = 2*(ng*n_ions + i);
            // Due to differing phase conventions in Python/C, as gvec_phases
            // was precalculated in Python, must use the complex conj
            cmult_conj((q_phases + 2*i), (gvec_phases + idx), gq_phase_ri);
            for (j = i; j < n_ions; j++) {
                idx = 2*(ng*n_ions + j);
                cmult_conj((q_phases + 2*j), (gvec_phases + idx), gq_phase_rj);
                // To divide by gq_phase_rj, multiply by complex conj
                cmult_conj(gq_phase_ri, gq_phase_rj, gq_phase_rij);
                for (a = 0; a < 3; a++) {
                    for (b = 0; b < 3; b++) {
                        idx = 2*(3*(3*i + a)*n_ions + 3*j + b);
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
    for (i = 0; i < n_ions; i++) {
        for (j = i; j < n_ions; j++) {
            memset(corr_tmp, 0, 18*sizeof(double));
            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    for (aa = 0; aa < 3; aa++) {
                        for (bb = 0; bb < 3; bb++) {
                            idx = 2*(3*(3*i + aa)*n_ions + 3*j + bb);
                            born_fac = born[9*i + 3*a + aa]*born[9*j + 3*b + bb];
                            corr_tmp[6*a + 2*b] += born_fac*corr[idx];
                            corr_tmp[6*a + 2*b + 1] += born_fac*corr[idx + 1];
                        }
                    }
                }
            }

            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    idx = 2*(3*(3*i + a)*n_ions + 3*j + b);
                    corr[idx] = corr_tmp[6*a + 2*b];
                    corr[idx + 1] = corr_tmp[6*a + 2*b + 1];
                }
            }

        }

        // Subtract q=0 correction from diagonal
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++) {
                idx = 2*(3*(3*i + a)*n_ions + 3*i + b);
                corr[idx] -= dipole_q0[18*i + 6*a + 2*b];
                corr[idx + 1] -= dipole_q0[18*i + 6*a + 2*b + 1];
            }
        }
    }

}

void calculate_gamma_correction(const double q_dir[3], const int n_ions,
    const double *cell_vec, const double *born, const double *dielectric,
    double *corr) {

    int i, j, a, b, idx;
    double fac, denominator;
    double *q_born_sum;

    denominator = 0;
    for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) {
           denominator += dielectric[3*a + b]*q_dir[a]*q_dir[b];
        }
    }
    fac = (4*PI)/(cell_volume(cell_vec)*denominator);

    q_born_sum = (double*)calloc(3*n_ions, sizeof(double));
    memset(q_born_sum, 0, 3*n_ions*sizeof(double));
    for (i = 0; i < n_ions; i++) {
        for (a = 0; a < 3; a++) {
            for (b = 0; b < 3; b++) {
                q_born_sum[3*i + a] += born[9*i + 3*a + b]*q_dir[b];
            }
        }
    }

    for (i = 0; i < n_ions; i++) {
        for (j = i; j < n_ions; j++) {
            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    idx = 2*(3*(3*i + a)*n_ions + 3*j + b);
                    corr[idx] = fac*q_born_sum[3*i + a]*q_born_sum[3*j + b];
                }
            }
        }
    }
    free((void*)q_born_sum);
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

int diagonalise_dyn_mat_zheevd(const int n_ions, const double qpt[3],
    double* dyn_mat, double* eigenvalues, ZheevdFunc zheevdptr) {

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

void evals_to_freqs(const int n_ions, double *eigenvalues) {
    int i;
    double tmp;
    for (i = 0; i < 3*n_ions; i++) {
        // Set imaginary frequencies to negative
        tmp = copysign(sqrt(fabs(eigenvalues[i])), eigenvalues[i]);
        eigenvalues[i] = tmp;
    }
}
