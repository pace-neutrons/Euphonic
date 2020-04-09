import os
from ..utils import get_data_path


def get_phonon_file():
    return os.path.join(get_data_path(), "NaH.phonon")
