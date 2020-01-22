#ifndef __py_util_H__
#define __py_util_H__

int attr_from_pyobj(PyObject *obj, const char *attr_name, PyObject **result);
int int_from_pyobj(PyObject *obj, const char *attr_name, int *result);
int double_from_pyobj(PyObject *obj, const char *attr_name, double *result);

#endif
