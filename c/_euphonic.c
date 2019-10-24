#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "dyn_mat.h"

static PyObject *calculate_dyn_mats(PyObject *self, PyObject *args) {
    // Define input args
    PyArrayObject *py_rqpts;
    PyArrayObject *py_fc;
    PyArrayObject *py_n_sc_ims;
    PyArrayObject *py_sc_im_idx;
    PyArrayObject *py_cell_ogs;
    PyArrayObject *py_sc_ogs;
//    PyArrayObject *py_ac_i;
//    PyArrayObject *py_g_evals;
//    PyArrayObject *py_g_evecs;
//    PyArrayObject *py_dmat_weighting;
//    int dipole;
//    char *asr;
//    int splitting;
    PyArrayObject *py_dmats;
    int nthreads = 1;

    // Define pointers to Python array data
    double *rqpts;
    double *fc;
    int *n_sc_ims;
    int *sc_im_idx;
    int *cell_ogs;
    int *sc_ogs;
//    int *ac_i;
//    double *g_evals;
//    double *g_evecs;
//    double *dmat_weighting;
    double *dmats;

    // Other vars
    int nions;
    int ncells;
    int nqpts;
    int nsc;
    int q;
    int maxims;
    double *qpt;
    double *dmat;
    int dmat_elems;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!i",
                          &PyArray_Type, &py_rqpts,
                          &PyArray_Type, &py_fc,
                          &PyArray_Type, &py_n_sc_ims,
                          &PyArray_Type, &py_sc_im_idx,
                          &PyArray_Type, &py_cell_ogs,
                          &PyArray_Type, &py_sc_ogs,
//                          &PyArray_Type, &py_ac_i,
//                          &PyArray_Type, &py_g_evals,
//                          &PyArray_Type, &py_g_evecs,
//                          &PyArray_Type, &py_dmat_weighting,
//                          &dipole,
//                          &asr,
//                          &splitting,
                          &PyArray_Type, &py_dmats,
                          &nthreads)) {
        return NULL;
    }

    rqpts = (double*) PyArray_DATA(py_rqpts);
    fc = (double*) PyArray_DATA(py_fc);
    n_sc_ims = (int*) PyArray_DATA(py_n_sc_ims);
    sc_im_idx = (int*) PyArray_DATA(py_sc_im_idx);
    cell_ogs = (int*) PyArray_DATA(py_cell_ogs);
    sc_ogs = (int*) PyArray_DATA(py_sc_ogs);
//    ac_i = (int*) PyArray_DATA(py_ac_i);
//    g_evals = (double*) PyArray_DATA(py_g_evals);
//    g_evecs = (double*) PyArray_DATA(py_g_evecs);
//    dmat_weighting = (double*) PyArray_DATA(py_dmat_weighting);
    dmats = (double*) PyArray_DATA(py_dmats);

    nions = PyArray_DIMS(py_fc)[1]/3;
    ncells = PyArray_DIMS(py_fc)[0];
    nqpts = PyArray_DIMS(py_rqpts)[0];
    nsc = PyArray_DIMS(py_sc_ogs)[0];
    maxims = PyArray_DIMS(py_sc_im_idx)[3];

    dmat_elems = 2*9*nions*nions;

    omp_set_num_threads(nthreads);
    #pragma omp parallel for
    for (q = 0; q < nqpts; q++) {
        qpt = (rqpts + 3*q);
        dmat = (dmats + q*dmat_elems);
        calculate_dyn_mat_at_q(qpt, nions, ncells, nsc, maxims, n_sc_ims, sc_im_idx,
            cell_ogs, sc_ogs, fc, dmat);
    }

    return Py_None;

}

static PyMethodDef _euphonic_methods[] = {
    {"calculate_dyn_mats", calculate_dyn_mats, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

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