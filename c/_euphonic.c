#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL EUPHONIC_NPY_ARRAY_API
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "dyn_mat.h"
#include "util.h"

#ifdef _WIN32
#include <Windows.h>
typedef void (__cdecl *LibFunc)(char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, double* rwork, int* lrwork,
    int* iwork, int* liwork, int* info);
#else
#include <dlfcn.h>
#include <dirent.h>
#include <glob.h>
#endif

static PyObject *calculate_phonons(PyObject *self, PyObject *args) {

    // Define input args
    PyObject *py_idata; // InterpolationData instance
    PyArrayObject *py_rqpts;
    PyArrayObject *py_fc;
    PyArrayObject *py_sc_ogs;
    PyArrayObject *py_asr_correction;
    PyArrayObject *py_dmat_weighting;
    PyArrayObject *py_dmats;
    PyArrayObject *py_evals;
    int reciprocal_asr;
    int nthreads = 1;
    const char *scipydir;

    // Define vars to be obtained from InterpolationData attributes
    int nions;
    PyArrayObject *py_n_sc_ims;
    PyArrayObject *py_sc_im_idx;
    PyArrayObject *py_cell_ogs;

    // Define pointers to Python array data
    double *rqpts;
    double *fc;
    int *sc_ogs;
    double *asr_correction;
    double *dmat_weighting;
    double *dmats;
    double *evals;
    int *n_sc_ims;
    int *sc_im_idx;
    int *cell_ogs;

    // Other vars
    int ncells;
    int nqpts;
    int q;
    int max_ims;
    int dmat_elems;
    int info;

    // Parse inputs
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!iO!O!is",
                          &py_idata,
                          &PyArray_Type, &py_rqpts,
                          &PyArray_Type, &py_fc,
                          &PyArray_Type, &py_sc_ogs,
                          &PyArray_Type, &py_asr_correction,
                          &PyArray_Type, &py_dmat_weighting,
                          &reciprocal_asr,
                          &PyArray_Type, &py_dmats,
                          &PyArray_Type, &py_evals,
                          &nthreads,
                          &scipydir)) {
        return NULL;
    }

    // Get rest of vars from InterpolationData object
    if(int_from_pyobj(py_idata, "n_ions", &nions) ||
        attr_from_pyobj(py_idata, "_n_sc_images", &py_n_sc_ims) ||
        attr_from_pyobj(py_idata, "_sc_image_i", &py_sc_im_idx) ||
        attr_from_pyobj(py_idata, "cell_origins", &py_cell_ogs)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to read attributes from object\n");
            return NULL;
    }

    // Load LAPACK funcs
#ifdef _WIN32
    LibFunc zheevd;
    HMODULE lib;
    WIN32_FIND_DATA filedata;
    HANDLE hfile;
    const char *libdir = "\\..\\..\\..\\scipy\\extra-dll\\";
    const char *fileglob = "libopenblas*dll";
    char buf[300];
    snprintf(buf, sizeof(buf), "%s%s%s", scipydir, libdir, fileglob);
    hfile = FindFirstFile(buf, &filedata);
    if (hfile == INVALID_HANDLE_VALUE) {
        PyErr_Format(PyExc_FileNotFoundError, "Could not find %s\n", buf);
        return NULL;
    }
    lib = LoadLibrary(filedata.cFileName);
    if (lib == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not load lib handle %s\n",
                     filedata.cFileName);
        return NULL;
    }
    zheevd = (LibFunc) GetProcAddress(lib, "zheevd_");
    if (zheevd == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not find zheevd_ in %s\n",
                     filedata.cFileName);
        return NULL;
#else
    void *lib;
    void (*zheevd)(char* jobz, char* uplo, int* n, double* a, int* lda,
        double* w, double* work, int* lwork, double* rwork, int* lrwork,
        int* iwork, int* liwork, int* info);
    const char *libdir = "/linalg";
    const char *fileglob = "/_flapack*so";
    glob_t globres;
    char buf[300];
    DIR *dir;

    snprintf(buf, sizeof(buf), "%s%s", scipydir, libdir);
    dir = opendir(buf);
    if (dir == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not open dir %s\n", buf);
        return NULL;
    }
    while (readdir(dir) != NULL) {
        snprintf(buf, sizeof(buf), "%s%s%s", scipydir, libdir, fileglob);
        glob(buf, 0, NULL, &globres);
        if (globres.gl_pathc > 0) {
            break;
        }
    }
    closedir(dir);
    if (globres.gl_pathc == 0) {
        PyErr_Format(PyExc_RuntimeError, "Glob failed: couldn't find %s\n", buf);
        return NULL;
    }

    snprintf(buf, sizeof(buf), "%s/%s", buf, globres.gl_pathv[0]);
    lib = dlopen(buf, RTLD_LAZY);
    if (lib == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not load lib handle %s\n", buf);
        return NULL;
    }
    zheevd = dlsym(lib, "zheevd_");
    if (zheevd == NULL) {
        PyErr_Format(PyExc_RuntimeError, "Could not find zheevd_ in %s\n", buf);
        return NULL;
    }
#endif

    // Point to Python array data
    rqpts = (double*) PyArray_DATA(py_rqpts);
    fc = (double*) PyArray_DATA(py_fc);
    sc_ogs = (int*) PyArray_DATA(py_sc_ogs);
    asr_correction = (double*) PyArray_DATA(py_asr_correction);
    dmat_weighting = (double*) PyArray_DATA(py_dmat_weighting);
    dmats = (double*) PyArray_DATA(py_dmats);
    evals = (double*) PyArray_DATA(py_evals);
    ncells = PyArray_DIMS(py_fc)[0];
    nqpts = PyArray_DIMS(py_rqpts)[0];
    n_sc_ims = (int*) PyArray_DATA(py_n_sc_ims);
    sc_im_idx = (int*) PyArray_DATA(py_sc_im_idx);
    cell_ogs = (int*) PyArray_DATA(py_cell_ogs);
    max_ims = PyArray_DIMS(py_sc_im_idx)[3];
    dmat_elems = 2*9*nions*nions;

    omp_set_num_threads(nthreads);
    #pragma omp parallel for
    for (q = 0; q < nqpts; q++) {
        double *qpt, *dmat, *eval;
        qpt = (rqpts + 3*q);
        dmat = (dmats + q*dmat_elems);
        eval = (evals + q*3*nions);

        calculate_dyn_mat_at_q(qpt, nions, ncells, max_ims, n_sc_ims, sc_im_idx,
            cell_ogs, sc_ogs, fc, dmat);

        if (reciprocal_asr) {
            add_arrays(dmat_elems, asr_correction, dmat);
        }

        mass_weight_dyn_mat(dmat_weighting, nions, dmat);

        info = diagonalise_dyn_mat_zheevd(nions, dmat, eval, zheevd);
        if (info != 0) {
            printf("INFO: Zheevd diagonalisation failed with info %i at "
                   "q-point %f %f %f\n", info, qpt[0], qpt[1], qpt[2]);
        }
        evals_to_freqs(nions, eval);
    }

#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif

    return Py_None;
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
