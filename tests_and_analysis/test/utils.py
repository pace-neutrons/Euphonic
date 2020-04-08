import os


def get_data_path() -> str:
    """
    Get the path to the data for the tests.

    Returns
    -------
    str: The path to the test data.
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def mock_has_method_call(mock, call, *args, **kwargs):
    """
    Returns true if the method name defined by the parameter call has been called on the unittest mock
      object defined by the parameter mock.

    If args and kwargs are defined, true is only returned if their values match the calls args and kwargs.

    Parameters
    ----------
    mock : unittest.mock.Mock
        The mock we are testing to see if it has been called with the call parameter
    call : str
        The method name we are testing to see has been called

    Returns
    -------
    True if the method call has been called on the mock, False if not.
    """
    for name, call_args, call_kwargs in mock.method_calls:
        print(name)
        if name == call and \
           all(call_args[index] == value for index, value in enumerate(args)) and \
           all(call_kwargs[key] == value for key, value in kwargs.items()):
            return True
    return False
