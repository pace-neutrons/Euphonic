import os
import pytest

if __name__ == "__main__":

    test_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(test_dir, "reports")
    if not os.path.exists(reports_dir):
        os.mkdir(reports_dir)

    os.chdir(reports_dir)
    test_exit_code = pytest.main([
        test_dir,
        "--benchmark-autosave",
        "--benchmark-histogram"
    ])
    os.chdir("..")
