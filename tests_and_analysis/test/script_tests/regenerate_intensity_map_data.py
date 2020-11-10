import json
from unittest.mock import patch
import euphonic.cli.intensity_map
from tests_and_analysis.test.script_tests.utils import (
    get_force_constants_file, get_current_plot_image_data,
    get_intensity_map_params, get_intensity_map_data_file)


@patch("matplotlib.pyplot.show")
def regenerate_intensity_map_data(_):

    json_data = {}

    for intensity_map_params in get_intensity_map_params():
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.intensity_map.main([get_force_constants_file()]
                                        + intensity_map_params)

        # Retrieve with gcf and write to file
        json_data[" ".join(intensity_map_params)
                  ] = get_current_plot_image_data()

    with open(get_intensity_map_data_file(), "w+") as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    regenerate_intensity_map_data()
