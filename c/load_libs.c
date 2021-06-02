#include <stdio.h>
#include <Python.h>
#include "load_libs.h"

// Global variable to cache pointer location
void *ZHEEVD_POINTER = NULL;

ZheevdFunc get_zheevd() {
    if (!ZHEEVD_POINTER)
    {
        // Take the Global interpreter lock to import scipy.linalg
        PyGILState_STATE gstate = PyGILState_Ensure();
        // scipy exposes pointers to all the BLAS and LAPACK functions in the
        // scipy.linalg.cython_[blas,lapack] submodules.
        // It's meant for use in cython but we can also use it here.
        PyObject *scipylinalg = PyImport_ImportModule("scipy.linalg.cython_lapack");
        if (!scipylinalg) {
            printf("Error: could not load the scipy.linalg module.");
            return NULL;
        }
        PyObject *pyx_capi = PyObject_GetAttrString(scipylinalg, "__pyx_capi__");
        if (!pyx_capi || !PyDict_Check(pyx_capi)) {
            printf("Error: could not load the C-api functions from scipy.linalg.");
            Py_DECREF(scipylinalg);
            return NULL;
        }
        PyObject *zheevdcapsule = PyDict_GetItemString(pyx_capi, "zheevd");
        if (!zheevdcapsule) {
            printf("Error: could not load the zheevd function from the cython api.");
            Py_DECREF(scipylinalg);
            Py_DECREF(pyx_capi);
            return NULL;
        }

        const char *name = PyCapsule_GetName(zheevdcapsule);
        ZHEEVD_POINTER = PyCapsule_GetPointer(zheevdcapsule, name);
        Py_DECREF(scipylinalg);
        Py_DECREF(pyx_capi);

        PyGILState_Release(gstate);
    }
    return ZHEEVD_POINTER;
}
