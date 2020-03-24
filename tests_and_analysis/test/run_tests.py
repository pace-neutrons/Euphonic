import sys
import os
import pytest

if __name__ == "__main__":

    # Set output directory to the reports directory under this file's directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    xml_filepath = os.path.join(test_dir, "reports", "junit_report.xml")
    test_exit_code = pytest.main(["-x", test_dir, "--junitxml={}".format(xml_filepath)])

    # Exit with a failure code if there are any errors or failures
    sys.exit(test_exit_code)
