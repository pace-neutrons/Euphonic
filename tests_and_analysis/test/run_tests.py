import sys
import os
import pytest

if __name__ == "__main__":

    # Set output file to the reports directory under this file's directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    xml_filedir = os.path.join(test_dir, "reports")
    os.mkdir(xml_filedir)

    # We may have reports from multiple platforms (windows, linux etc.)
    filename_prefix = "junit_report"
    filenum = 0
    for filename in os.listdir(xml_filedir):
        if filename_prefix in filename:
            filenum += 1
    xml_filepath = os.path.join(test_dir, "reports", "{}{}.xml".format(filename_prefix, filenum))

    # Run tests and get the resulting exit code
    # 0 is success, 1-5 are different forms of failure (see pytest docs for details)
    test_exit_code = pytest.main(["-x", test_dir, "--junitxml={}".format(xml_filepath)])

    # Exit with a failure code if there are any errors or failures
    sys.exit(test_exit_code)
