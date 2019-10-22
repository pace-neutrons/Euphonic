#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *calculate_phonons_at_q(PyObject *self, PyObject *args) {
    // Define args
    PyArrayObject *py_reduced_qpts;
    PyArrayObject *py_qpts_i;
    PyArrayObject *py_fc_img_weighted;
    PyArrayObject *py_n_sc_images;
    PyArrayObject *py_sc_image_i;
    PyArrayObject *py_cell_origins;
    PyArrayObject *py_sc_image_r;
    PyArrayObject *py_ac_i;
    PyArrayObject *py_g_evals;
    PyArrayObject *py_g_evecs;
    PyArrayObject *py_dyn_mat_weighting;
    int dipole;
    char *asr;
    int splitting;

    // Define pointers to array data
    double *reduced_qpts;
    int *qpts_i;
    double *fc_img_weighted;
    int *n_sc_images;
    int *sc_image_i;
    int *cell_origins;
    int *sc_image_r;
    int *ac_i;
    double *g_evals;
    double *g_evecs;
    double *dyn_mat_weighting;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!isi",
//    if (!PyArg_ParseTuple(args, "O!",
                          &PyArray_Type, &py_reduced_qpts,
                          &PyArray_Type, &py_qpts_i,
                          &PyArray_Type, &py_fc_img_weighted,
                          &PyArray_Type, &py_n_sc_images,
                          &PyArray_Type, &py_sc_image_i,
                          &PyArray_Type, &py_cell_origins,
                          &PyArray_Type, &py_sc_image_r,
                          &PyArray_Type, &py_ac_i,
                          &PyArray_Type, &py_g_evals,
                          &PyArray_Type, &py_g_evecs,
                          &PyArray_Type, &py_dyn_mat_weighting,
                          &dipole,
                          &asr,
                          &splitting)) {
        return NULL;
    }

    reduced_qpts = (double*) PyArray_DATA(py_reduced_qpts);
    qpts_i = (int*) PyArray_DATA(py_qpts_i);
    fc_img_weighted = (double*) PyArray_DATA(py_fc_img_weighted);
    n_sc_images = (int*) PyArray_DATA(py_n_sc_images);
    sc_image_i = (int*) PyArray_DATA(py_sc_image_i);
    cell_origins = (int*) PyArray_DATA(py_cell_origins);
    sc_image_r = (int*) PyArray_DATA(py_sc_image_r);
    ac_i = (int*) PyArray_DATA(py_ac_i);
    g_evals = (double*) PyArray_DATA(py_g_evals);
    g_evecs = (double*) PyArray_DATA(py_g_evecs);
    dyn_mat_weighting = (double*) PyArray_DATA(py_dyn_mat_weighting);

    return Py_None;

}

static PyMethodDef _euphonic_methods[] = {
    {"calculate_phonons_at_q", calculate_phonons_at_q, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _euphonic_module_def = {
    PyModuleDef_HEAD_INIT,
//    "_euphonic",
    "euphonic_c",
    NULL,
    -1,
    _euphonic_methods
};

//PyMODINIT_FUNC PyInit__euphonic(void) {
PyMODINIT_FUNC PyInit_euphonic_c(void) {
    import_array();
    return PyModule_Create(&_euphonic_module_def);
}