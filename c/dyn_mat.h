#ifndef __dyn_mat_H__
#define __dyn_mat_H__

void calculate_dyn_mat_at_q(double *qpt, int n_ions, int n_cells, int n_sc,
    int max_ims, int *n_sc_images, int *sc_image_i, int *cell_origins,
    int *sc_origins, double *fc_mat, double *dyn_mat);

#endif