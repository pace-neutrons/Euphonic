import os
from euphonic.data.bands import BandsData
from euphonic.data.phonon import PhononData
from argparse import ArgumentParser
from typing import List


def load_data_from_file(filename):
    path, file = os.path.split(filename)
    seedname = file[:file.rfind('.')]
    if file.endswith('.bands'):
        data = BandsData.from_castep(seedname, path=path)
    else:
        data = PhononData.from_castep(seedname, path=path)
    return data, seedname, file


def get_args_and_set_up_and_down(parser: ArgumentParser, params: List[str] = None):
    if params is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(params)
    # If neither -up nor -down specified, plot both
    if not args.up and not args.down:
        args.up = True
        args.down = True
    return args


def matplotlib_save_or_show(save_filename: str = None):
    import matplotlib.pyplot as plt
    if save_filename is not None:
        plt.savefig(save_filename)
    else:
        plt.show()
