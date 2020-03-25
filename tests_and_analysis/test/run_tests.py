import sys
import os
import pytest
import coverage

if __name__ == "__main__":

    # Set output directory to the reports directory under this file's directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    xml_dir = os.path.join(test_dir, "reports")
    junit_xml_filepath = os.path.join(xml_dir, "junit_report.xml")
    coverage_xml_filepath = os.path.join(xml_dir, "coverage.xml")

    # Start recording coverage
    cov = coverage.Coverage(config_file=os.path.join(test_dir, ".coveragerc"))
    cov.start()

    # Run tests and get the resulting exit code
    # 0 is success, 1-5 are different forms of failure (see pytest docs for details)
    test_exit_code = pytest.main(["-x", test_dir, "--junitxml={}".format(junit_xml_filepath)])

    # Report coverage
    cov.stop()
    cov.xml_report(outfile=coverage_xml_filepath)

    # Exit with a failure code if there are any errors or failures
    sys.exit(test_exit_code)
