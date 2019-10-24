#ifndef __dyn_mat_H__
#define __dyn_mat_H__

void calculate_dyn_mat_at_q(const double *qpt, const int n_ions,
    const int n_cells, const int n_sc, const int max_ims,
    const int *n_sc_images, const int *sc_image_i, const int *cell_origins,
    const int *sc_origins, const double *fc_mat, double *dyn_mat);

#endif