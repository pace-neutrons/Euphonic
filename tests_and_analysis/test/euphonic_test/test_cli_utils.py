"""Additional tests of euphonic.cli.utils internals"""


import pytest

from euphonic.cli.utils import _get_q_distance
from euphonic.ureg import Quantity


def test_get_q_distance():
    """Test private function _get_q_distance"""

    assert _get_q_distance("mm", 1) == Quantity(1, "1 / mm")

    with pytest.raises(ValueError, match="Length unit not known"):
        _get_q_distance("elephant", 4)
