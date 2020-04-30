import os
import pytest

if __name__ == "__main__":

    test_dir = os.path.dirname(os.path.abspath(__file__))

    test_exit_code = pytest.main([
        test_dir,
        "--junitxml=reports/junit_report.xml"
    ])
