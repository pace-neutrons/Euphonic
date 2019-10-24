#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <math.h>

#define PI 3.14159265358979323846

void calculate_dyn_mat_at_q(const double *qpt, const int n_ions,
    const int n_cells, const int n_sc, const int max_ims,
    const int *n_sc_images, const int *sc_image_i, const int *cell_origins,
    const int *sc_origins, const double *fc_mat, double *dyn_mat) {

    int i, j, n, nc, k, sc, ii, jj, idx, fc_elems;
    double qdotr;
    double phase_r;
    double phase_i;
    double tmp;

    // Array strides
    int s_n[2] = {n_ions*n_ions, n_ions}; // For n_sc_images
    int s_i[3] = {n_ions*n_ions*max_ims, n_ions*max_ims, max_ims}; // For sc_image_i
    int s_fc = 9*n_ions*n_ions; // For fc_mat

    for (i = 0; i < n_ions; i++) {
        for (j = 0; j < n_ions; j++) {
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
                    phase_i += sin(2*PI*qdotr);
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