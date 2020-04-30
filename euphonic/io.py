import json
import numpy as np


def _ndarray_to_list(dictionary):
    """
    For a dictionary, convert all ndarray key values to lists
    """
    for key, val in dictionary.items():
        if isinstance(val, np.ndarray):
            dictionary[key] = val.tolist()
    return dictionary


def _list_to_ndarray(dictionary):
    """
    For a dictionary, convert all list key values to ndarrays
    """
    for key, val in dictionary.items():
        if isinstance(val, list):
            dictionary[key] = np.array(val)
    return dictionary

def _obj_to_json_file(obj, filename):
    """
    Generic function for writing to a JSON file from a Euphonic object
    """
    with open(filename, 'w') as f:
        dout = _ndarray_to_list(obj.to_dict())
        json.dump(dout, f, indent=4, sort_keys=True)
    print('Written to ' + filename)


def _obj_from_json_file(cls, filename):
    """
    Generic function for reading from a JSON file to a Euphonic object
    """
    with open(filename, 'r') as f:
        obj_dict = json.loads(f.read())
    return cls.from_dict(obj_dict)
