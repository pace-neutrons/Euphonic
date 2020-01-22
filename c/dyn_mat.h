#ifndef __dyn_mat_H__
#define __dyn_mat_H__

void calculate_dyn_mat_at_q(const double *qpt, const int n_ions,
    const int n_cells, const int max_ims, const int *n_sc_images,
    const int *sc_image_i, const int *cell_origins, const int *sc_origins,
    const double *fc_mat, double *dyn_mat);

void calculate_dipole_correction(const double *qpt, const int n_ions,
    const double *cell_vec, const double *recip, const double *ion_r,
    const double *born, const double *dielectric, const double *H_ab,
    const double *cells, const int n_dcells, const double *gvec_phases,
    const double *gvecs_cart, const int n_gvecs, const double *dipole_q0,
    const double eta, double *corr);

void calculate_gamma_correction(const double q_dir[3], const int n_ions,
    const double *cell_vec, const double *born, const double *dielectric,
    double *corr);

void mass_weight_dyn_mat(const double *dyn_mat_weighting, const int n_ions,
    double *dyn_mat);

int diagonalise_dyn_mat_zheevd(const int n_ions, const double qpt[3],
    double *dyn_mat, double *eigenvalues, ZheevdFunc zheevdptr);

void evals_to_freqs(const int n_ions, double *eigenvalues);

#endif
