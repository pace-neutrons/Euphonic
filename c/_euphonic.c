#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL EUPHONIC_NPY_ARRAY_API
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "load_libs.h"
#include "dyn_mat.h"
#include "py_util.h"
#include "util.h"

static PyObject *calculate_phonons(PyObject *self, PyObject *args) {

    // Define input args
    PyObject *py_idata; // ForceConstants instance
    PyArrayObject *py_cell_vec;
    PyArrayObject *py_recip_vec;
    PyArrayObject *py_rqpts;
    PyArrayObject *py_split_idx;
    PyArrayObject *py_q_dirs;
    PyArrayObject *py_fc;
    PyArrayObject *py_sc_ogs;
    PyArrayObject *py_asr_correction;
    PyArrayObject *py_dmat_weighting;
    PyArrayObject *py_evals;
    PyArrayObject *py_dmats;
    PyArrayObject *py_modegs;
    PyArrayObject *py_all_ogs_cart;
    int dipole;
    int splitting;
    int n_threads = 1;

    // Define vars to be obtained from ForceConstants attributes
    PyObject *py_crystal; // Crystal object
    PyArrayObject *py_n_sc_ims;
    PyArrayObject *py_sc_im_idx;
    PyArrayObject *py_cell_ogs;
    // Extra vars only required if dipole = True
    PyArrayObject *py_born;
    PyArrayObject *py_dielectric;
    PyDictObject *py_dipole_init_data;
    double lambda;
    PyArrayObject *py_H_ab;
    PyArrayObject *py_dipole_cells;
    PyArrayObject *py_gvec_phases;
    PyArrayObject *py_gvecs_cart;
    PyArrayObject *py_dipole_q0;

    // Vars to be obtained from Crystal attributes
    int n_atoms;
    PyArrayObject *py_atom_r;

    // Define pointers to Python array data
    double *cell_vec;
    double *recip_vec;
    double *rqpts;
    int *split_idx;
    double *q_dirs;
    double *fc;
    int *sc_ogs;
    double *asr_correction;
    double *dmat_weighting;
    double *evals;
    double *dmats;
    double *modegs;
    double *all_ogs_cart;
    int *n_sc_ims;
    int *sc_im_idx;
    int *cell_ogs;
    // Extra vars only required if dipole = True
    double *atom_r;
    double *born;
    double *dielectric;
    double *H_ab;
    double *dipole_cells;
    double *gvec_phases;
    double *gvecs_cart;
    double *dipole_q0;

    // Other vars
    int n_cells;
    int n_rqpts;
    int dmats_len;
    int modegs_len;
    int asr_corr_len;
    int n_split_qpts;
    int q, i, qpos;
    int max_ims;
    int dmat_elems;
    // Extra vars only required if dipole = True
    int n_dipole_cells;
    int n_gvecs;

    // Parse inputs
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!O!O!O!iiO!O!O!O!i",
                          &py_idata,
                          &PyArray_Type, &py_cell_vec,
                          &PyArray_Type, &py_recip_vec,
                          &PyArray_Type, &py_rqpts,
                          &PyArray_Type, &py_split_idx,
                          &PyArray_Type, &py_q_dirs,
                          &PyArray_Type, &py_fc,
                          &PyArray_Type, &py_sc_ogs,
                          &PyArray_Type, &py_asr_correction,
                          &PyArray_Type, &py_dmat_weighting,
                          &dipole,
                          &splitting,
                          &PyArray_Type, &py_evals,
                          &PyArray_Type, &py_dmats,
                          &PyArray_Type, &py_modegs,
                          &PyArray_Type, &py_all_ogs_cart,
                          &n_threads)) {
        return NULL;
    }
    // Load library functions
    // Load before calling PyObject_GetAttrString so we don't have to
    // py_DECREF if library loading fails
    ZheevdFunc zheevd;
    zheevd = get_zheevd();
    if (zheevd == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not load zheevd function\n");
        return NULL;
    }

    // Get rest of vars from ForceConstants object
    if (attr_from_pyobj(py_idata, "crystal", &py_crystal) ||
        attr_from_pyobj(py_idata, "_n_sc_images", &py_n_sc_ims) ||
        attr_from_pyobj(py_idata, "_sc_image_i", &py_sc_im_idx) ||
        attr_from_pyobj(py_idata, "cell_origins", &py_cell_ogs)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to read attributes from object\n");
            return NULL;
    }
    if (dipole) {
        if (attr_from_pyobj(py_idata, "_dipole_init_data", &py_dipole_init_data)) {
                PyErr_Format(PyExc_RuntimeError,
                             "Failed to read dipole init data from object\n");
                return NULL;
        }
        if (attr_from_pyobj(py_idata, "_born", &py_born) ||
            attr_from_pyobj(py_idata, "_dielectric", &py_dielectric) ||
            double_from_pydict(py_dipole_init_data, "lambda", &lambda) ||
            val_from_pydict(py_dipole_init_data, "H_ab", &py_H_ab) ||
            val_from_pydict(py_dipole_init_data, "cells", &py_dipole_cells) ||
            val_from_pydict(py_dipole_init_data, "gvec_phases", &py_gvec_phases) ||
            val_from_pydict(py_dipole_init_data, "gvecs_cart", &py_gvecs_cart) ||
            val_from_pydict(py_dipole_init_data, "dipole_q0", &py_dipole_q0)) {
                PyErr_Format(PyExc_RuntimeError,
                             "Failed to read dipole attributes from object\n");
                return NULL;
        }
    }
    // Get vars from Crystal object
    if (int_from_pyobj(py_crystal, "n_atoms", &n_atoms) ||
        attr_from_pyobj(py_crystal, "atom_r", &py_atom_r)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to read attributes from Crystal object\n");
            return NULL;
    }

    // Point to Python array data
    cell_vec = (double*) PyArray_DATA(py_cell_vec);
    recip_vec = (double*) PyArray_DATA(py_recip_vec);
    rqpts = (double*) PyArray_DATA(py_rqpts);
    split_idx = (int*) PyArray_DATA(py_split_idx);
    q_dirs = (double*) PyArray_DATA(py_q_dirs);
    fc = (double*) PyArray_DATA(py_fc);
    sc_ogs = (int*) PyArray_DATA(py_sc_ogs);
    asr_correction = (double*) PyArray_DATA(py_asr_correction);
    dmat_weighting = (double*) PyArray_DATA(py_dmat_weighting);
    evals = (double*) PyArray_DATA(py_evals);
    dmats = (double*) PyArray_DATA(py_dmats);
    modegs = (double*) PyArray_DATA(py_modegs);
    all_ogs_cart = (double*) PyArray_DATA(py_all_ogs_cart);
    n_sc_ims = (int*) PyArray_DATA(py_n_sc_ims);
    sc_im_idx = (int*) PyArray_DATA(py_sc_im_idx);
    cell_ogs = (int*) PyArray_DATA(py_cell_ogs);
    n_cells = PyArray_DIMS(py_fc)[0];
    n_rqpts = PyArray_DIMS(py_rqpts)[0];
    n_split_qpts = PyArray_DIMS(py_split_idx)[0];
    dmats_len = PyArray_DIMS(py_dmats)[0];
    modegs_len = PyArray_DIMS(py_modegs)[0];
    asr_corr_len = PyArray_DIMS(py_asr_correction)[0];
    max_ims = PyArray_DIMS(py_sc_im_idx)[3];
    dmat_elems = 2*9*n_atoms*n_atoms;
    if (dipole) {
        atom_r = (double*) PyArray_DATA(py_atom_r);
        born = (double*) PyArray_DATA(py_born);
        dielectric = (double*) PyArray_DATA(py_dielectric);
        H_ab = (double*) PyArray_DATA(py_H_ab);
        dipole_cells = (double*) PyArray_DATA(py_dipole_cells);
        gvec_phases = (double*) PyArray_DATA(py_gvec_phases);
        gvecs_cart = (double*) PyArray_DATA(py_gvecs_cart);
        dipole_q0 = (double*) PyArray_DATA(py_dipole_q0);
        n_dipole_cells = PyArray_DIMS(py_dipole_cells)[0];
        n_gvecs = PyArray_DIMS(py_gvec_phases)[0];
    }



    omp_set_num_threads(n_threads);
    #pragma omp parallel
    {
        const bool calc_dmat_grad = (modegs_len > 0) ? true : false;
        double *corr, *dmat_per_q, *dmat_grad;
        if (dipole) {
            corr = (double*) malloc(dmat_elems*sizeof(double));
        }
        // If space for the eigenvectors has not been allocated, assume they
        // aren't to be returned and just allocate just enough memory for each
        // q-point calculation
        if (dmats_len == 0) {
            dmat_per_q = (double*) malloc(dmat_elems*sizeof(double));
        }
        // If space for the mode gradients has been allocated, assume they
        // should be calculated and allocate space for dyn mat gradients
        if (calc_dmat_grad) {
            dmat_grad = (double*) malloc(3*dmat_elems*sizeof(double));
        } else {
            dmat_grad = NULL;
        }
        #pragma omp for
        for (q = 0; q < n_rqpts; q++) {
            double *qpt, *dmat, *eval, *modeg;
            qpt = (rqpts + 3*q);
            eval = (evals + q*3*n_atoms);

            if (dmats_len == 0) {
                dmat = dmat_per_q;
            } else {
                dmat = (dmats + q*dmat_elems);
            }
            if (calc_dmat_grad) {
                modeg = (modegs + q*3*n_atoms*6);
            }
            calculate_dyn_mat_at_q(qpt, n_atoms, n_cells, max_ims, n_sc_ims,
                sc_im_idx, cell_ogs, sc_ogs, fc, all_ogs_cart, calc_dmat_grad,
                dmat, dmat_grad);

            if (dipole) {
                calculate_dipole_correction(qpt, n_atoms, cell_vec, recip_vec,
                    atom_r, born, dielectric, H_ab, dipole_cells,
                    n_dipole_cells, gvec_phases, gvecs_cart, n_gvecs,
                    dipole_q0, lambda, corr);
                add_arrays(dmat_elems, corr, dmat);
            }

            if (asr_corr_len > 0) {
                add_arrays(dmat_elems, asr_correction, dmat);
            }

            // Calculate non-analytical correction for LO-TO splitting
            if (splitting && is_gamma(qpt)) {
                // Find q-direction for this q-point
                qpos = -1;
                for (i = 0; i < n_split_qpts; i++) {
                    if (split_idx[i] == q) {
                        qpos = i;
                        break;
                    }
                }
                calculate_gamma_correction((q_dirs + 3*qpos), n_atoms,
                    cell_vec, recip_vec, born, dielectric, corr);
                add_arrays(dmat_elems, corr, dmat);
            }

            mass_weight_dyn_mat(dmat_weighting, n_atoms, 2, dmat);
            diagonalise_dyn_mat_zheevd(n_atoms, qpt, dmat, eval, zheevd);
            evals_to_freqs(n_atoms, eval);

            if (calc_dmat_grad) {
                mass_weight_dyn_mat(dmat_weighting, n_atoms, 6, dmat_grad);
                calculate_mode_gradients(n_atoms, eval, dmat, dmat_grad, modeg);
            }
        }
        if (dipole) {
            free((void*)corr);
        }
        if (dmats_len == 0) {
            free((void*)dmat_per_q);
        }
        if (calc_dmat_grad) {
            free((void*)dmat_grad);
        }
    }

    // PyObject_GetAttrString returns a "new" reference, need to decref
    Py_DECREF(py_crystal);
    Py_DECREF(py_n_sc_ims);
    Py_DECREF(py_sc_im_idx);
    Py_DECREF(py_cell_ogs);
    Py_DECREF(py_atom_r);
    if (dipole){
        // Note PyDict_GetItemString returns "borrowed" ref so don't need
        // to decref lambda, H_ab etc.
        Py_DECREF(py_born);
        Py_DECREF(py_dielectric);
        Py_DECREF(py_dipole_init_data);
    }

    Py_RETURN_NONE;
}

static PyMethodDef _euphonic_methods[] = {
    {"calculate_phonons", calculate_phonons, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef _euphonic_module_def = {
    PyModuleDef_HEAD_INIT,
    "_euphonic",
    NULL,
    -1,
    _euphonic_methods
};

PyMODINIT_FUNC PyInit__euphonic(void) {
    import_array();
    return PyModule_Create(&_euphonic_module_def);
}
#else
PyMODINIT_FUNC init_euphonic() {
    import_array();
    Py_InitModule3("_euphonic", _euphonic_methods, NULL);
}
#endif
