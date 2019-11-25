#ifndef __util_H__
#define __util_H__

int attr_from_pyobj(PyObject *obj, const char *attr_name, PyObject **result);
int int_from_pyobj(PyObject *obj, const char *attr_name, int *result);

#endif
