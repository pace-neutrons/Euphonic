#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL EUPHONIC_NPY_ARRAY_API
#include <Python.h>
#include <numpy/arrayobject.h>

int attr_from_pyobj(PyObject *obj, const char *attr_name, PyObject **result) {
/* Given a PyObject and the name of one of its attributes, get the address of
 * that attribute, and alter the pointer pointed to by result to point to that
 * address */
    if (PyObject_HasAttrString(obj, attr_name)) {
        PyObject *tmp = PyObject_GetAttrString(obj, attr_name);
        *result = tmp; 
    } else {
        printf("Object has no attr %s\n", attr_name);
        return 1;
    }
    return 0;
}

int int_from_pyobj(PyObject *obj, const char *attr_name, int *result) {
/* Given a PyObject and the name of one of its attributes, read an integer
 * from that attribute and store it in the address pointed to by result */
    PyObject *tmp;
    attr_from_pyobj(obj, attr_name, &tmp);
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check(tmp)) {
        *result = (int) PyLong_AsLong(tmp);
#else
    if (PyInt_Check(tmp)) {
        *result = (int) PyInt_AsLong(tmp);
#endif
    } else {
        printf("Incorrect type for %s\n", attr_name);
        return 1;
    }
    return 0;
}

int double_from_pyobj(PyObject *obj, const char *attr_name,
                         double *result) {
/* Given a PyObject and the name of one of its attributes, read a double
 * from that attribute and store it in the address pointed to by result */
    PyObject *tmp;
    attr_from_pyobj(obj, attr_name, &tmp);
    if (PyFloat_Check(tmp)) {
        *result = (double) PyFloat_AsDouble(tmp);
    } else {
        printf("Incorrect type for %s\n", attr_name);
        return 1;
    }
    return 0;
}
