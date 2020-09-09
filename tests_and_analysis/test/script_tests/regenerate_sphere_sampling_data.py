import json
from unittest.mock import patch
import numpy

import scripts.euphonic_sphere_sampling
from tests_and_analysis.test.script_tests.utils import (
    get_sphere_sampling_params, get_sphere_sampling_data_file,
    get_current_plot_offsets)


@patch("matplotlib.pyplot.show")
def regenerate_sphere_sampling_data(_):

    json_data = {}

    for sampling_params in get_sphere_sampling_params():
        # Reset random number generator for deterministic results
        numpy.random.seed(0)

        # Generate current figure for us to retrieve with gcf
        scripts.euphonic_sphere_sampling.main(sampling_params)

        # Retrieve with gcf and write to file
        json_data[" ".join(sampling_params)] = get_current_plot_offsets()

    with open(get_sphere_sampling_data_file(), "w+") as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    regenerate_sphere_sampling_data()
