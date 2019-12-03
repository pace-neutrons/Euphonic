#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL EUPHONIC_NPY_ARRAY_API
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "dyn_mat.h"
#include "py_util.h"
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
    PyArrayObject *py_qpts_i;
    PyArrayObject *py_fc;
    PyArrayObject *py_sc_ogs;
    PyArrayObject *py_asr_correction;
    PyArrayObject *py_dmat_weighting;
    PyArrayObject *py_evals;
    PyArrayObject *py_dmats;
    PyArrayObject *py_split_evals;
    PyArrayObject *py_split_evecs;
    PyArrayObject *py_split_i;
    int reciprocal_asr;
    int dipole;
    int nthreads = 1;
    const char *scipydir;

    // Define vars to be obtained from InterpolationData attributes
    int n_ions;
    PyArrayObject *py_n_sc_ims;
    PyArrayObject *py_sc_im_idx;
    PyArrayObject *py_cell_ogs;
    // Extra vars only required if dipole = True
    PyArrayObject *py_cell_vec;
    PyArrayObject *py_recip_vec;
    PyArrayObject *py_ion_r;
    PyArrayObject *py_born;
    PyArrayObject *py_dielectric;
    double eta;
    PyArrayObject *py_H_ab;
    PyArrayObject *py_dipole_cells;
    PyArrayObject *py_gvec_phases;
    PyArrayObject *py_gvecs_cart;
    PyArrayObject *py_dipole_q0;

    // Define pointers to Python array data
    double *rqpts;
    int *qpts_i;
    double *fc;
    int *sc_ogs;
    double *asr_correction;
    double *dmat_weighting;
    double *evals;
    double *dmats;
    double *split_evals;
    double *split_evecs;
    int *split_i;
    int *n_sc_ims;
    int *sc_im_idx;
    int *cell_ogs;
    // Extra vars only required if dipole = True
    double *cell_vec;
    double *recip_vec;
    double *ion_r;
    double *born;
    double *dielectric;
    double *H_ab;
    double *dipole_cells;
    double *gvec_phases;
    double *gvecs_cart;
    double *dipole_q0;

    // Other vars
    int n_cells;
    int nqpts;
    int q;
    int max_ims;
    int dmat_elems;
    int info;
    // Extra vars only required if dipole = True
    int n_dipole_cells;
    int n_gvecs;

    // Parse inputs
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!iiO!O!O!O!O!is",
                          &py_idata,
                          &PyArray_Type, &py_rqpts,
                          &PyArray_Type, &py_qpts_i,
                          &PyArray_Type, &py_fc,
                          &PyArray_Type, &py_sc_ogs,
                          &PyArray_Type, &py_asr_correction,
                          &PyArray_Type, &py_dmat_weighting,
                          &reciprocal_asr,
                          &dipole,
                          &PyArray_Type, &py_evals,
                          &PyArray_Type, &py_dmats,
                          &PyArray_Type, &py_split_evals,
                          &PyArray_Type, &py_split_evecs,
                          &PyArray_Type, &py_split_i,
                          &nthreads,
                          &scipydir)) {
        return NULL;
    }

    // Get rest of vars from InterpolationData object
    if (int_from_pyobj(py_idata, "n_ions", &n_ions) ||
        attr_from_pyobj(py_idata, "_n_sc_images", &py_n_sc_ims) ||
        attr_from_pyobj(py_idata, "_sc_image_i", &py_sc_im_idx) ||
        attr_from_pyobj(py_idata, "cell_origins", &py_cell_ogs)) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to read attributes from object\n");
            return NULL;
    }
    if (dipole) {
        if (attr_from_pyobj(py_idata, "_cell_vec", &py_cell_vec) ||
            attr_from_pyobj(py_idata, "_recip_vec", &py_recip_vec) ||
            attr_from_pyobj(py_idata, "ion_r", &py_ion_r) ||
            attr_from_pyobj(py_idata, "_born", &py_born) ||
            attr_from_pyobj(py_idata, "dielectric", &py_dielectric) ||
            double_from_pyobj(py_idata, "_eta", &eta) ||
            attr_from_pyobj(py_idata, "_H_ab", &py_H_ab) ||
            attr_from_pyobj(py_idata, "_cells", &py_dipole_cells) ||
            attr_from_pyobj(py_idata, "_gvec_phases", &py_gvec_phases) ||
            attr_from_pyobj(py_idata, "_gvecs_cart", &py_gvecs_cart) ||
            attr_from_pyobj(py_idata, "_dipole_q0", &py_dipole_q0)) {
                PyErr_Format(PyExc_RuntimeError,
                             "Failed to read dipole attributes from object\n");
                return NULL;
        }
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
    qpts_i = (int*) PyArray_DATA(py_qpts_i);
    fc = (double*) PyArray_DATA(py_fc);
    sc_ogs = (int*) PyArray_DATA(py_sc_ogs);
    asr_correction = (double*) PyArray_DATA(py_asr_correction);
    dmat_weighting = (double*) PyArray_DATA(py_dmat_weighting);
    evals = (double*) PyArray_DATA(py_evals);
    dmats = (double*) PyArray_DATA(py_dmats);
    split_evals = (double*) PyArray_DATA(py_split_evals);
    split_evecs = (double*) PyArray_DATA(py_split_evecs);
    split_i = (double*) PyArray_DATA(py_split_i);
    n_sc_ims = (int*) PyArray_DATA(py_n_sc_ims);
    sc_im_idx = (int*) PyArray_DATA(py_sc_im_idx);
    cell_ogs = (int*) PyArray_DATA(py_cell_ogs);
    n_cells = PyArray_DIMS(py_fc)[0];
    nqpts = PyArray_DIMS(py_rqpts)[0];
    max_ims = PyArray_DIMS(py_sc_im_idx)[3];
    dmat_elems = 2*9*n_ions*n_ions;
    if (dipole) {
        cell_vec = (double*) PyArray_DATA(py_cell_vec);
        recip_vec = (double*) PyArray_DATA(py_recip_vec);
        ion_r = (double*) PyArray_DATA(py_ion_r);
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

    omp_set_num_threads(nthreads);
    #pragma omp parallel
    {
        double *dipole_corr;
        if (dipole) {
            dipole_corr = (double*) malloc(dmat_elems*sizeof(double));
        }
        #pragma omp for
        for (q = 0; q < nqpts; q++) {
            double *qpt, *dmat, *eval;
            qpt = (rqpts + 3*q);
            dmat = (dmats + q*dmat_elems);
            eval = (evals + q*3*n_ions);

            calculate_dyn_mat_at_q(qpt, n_ions, n_cells, max_ims, n_sc_ims,
                sc_im_idx, cell_ogs, sc_ogs, fc, dmat);

            if (dipole) {
                calculate_dipole_correction(qpt, n_ions, cell_vec, recip_vec,
                    ion_r, born, dielectric, H_ab, dipole_cells,
                    n_dipole_cells, gvec_phases, gvecs_cart, n_gvecs,
                    dipole_q0, eta, dipole_corr);
                add_arrays(dmat_elems, dipole_corr, dmat);
            }

            if (reciprocal_asr) {
                add_arrays(dmat_elems, asr_correction, dmat);
            }

            mass_weight_dyn_mat(dmat_weighting, n_ions, dmat);

            info = diagonalise_dyn_mat_zheevd(n_ions, dmat, eval, zheevd);
            if (info != 0) {
                printf("INFO: Zheevd diagonalisation failed with info %i at "
                       "q-point %f %f %f\n", info, qpt[0], qpt[1], qpt[2]);
            }
            evals_to_freqs(n_ions, eval);
        }
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
