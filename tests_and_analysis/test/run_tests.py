import sys
import os
import pytest
import coverage
import argparse

if __name__ == "__main__":

    # Check whether the recording coverage has been requested
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", action="store_true",
                        help="If present report coverage in a coverage.xml file in reports")
    parser.add_argument("--report", action="store_true",
                        help="If present report test results to junit_report*.xml files")
    parser.add_argument("-t", "--test-file", dest="test_file", action="store",
                        help="The test file to run", default=None)
    args_parsed = parser.parse_args()
    do_record_coverage = args_parsed.cov
    do_report_results = args_parsed.report

    # Set output directory to the reports directory under this file's directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    xml_dir = os.path.join(test_dir, "reports")
    if not os.path.exists(xml_dir):
        os.mkdir(xml_dir)

    # Handle the test file to be executed
    if args_parsed.test_file is not None:
        tests = os.path.join(test_dir, args_parsed.test_file)
    else:
        tests = test_dir

    # Start recording coverage if requested
    cov = None
    if do_record_coverage:
        coveragerc_filepath = os.path.join(test_dir, ".coveragerc")
        cov = coverage.Coverage(config_file=coveragerc_filepath)
        cov.start()

    # Run tests and get the resulting exit code
    # 0 is success, 1-5 are different forms of failure (see pytest docs for details)
    if do_report_results:
        junit_xml_filepath = os.path.join(xml_dir, "junit_report.xml")
        test_exit_code = pytest.main(["-x", tests, "--junitxml={}".format(junit_xml_filepath)])
    else:
        test_exit_code = pytest.main(["-x", tests])

    # Report coverage if requested
    if do_record_coverage and cov is not None:
        cov.stop()
        coverage_xml_filepath = os.path.join(xml_dir, "coverage.xml")
        cov.xml_report(outfile=coverage_xml_filepath)

    # Exit with a failure code if there are any errors or failures
    sys.exit(test_exit_code)
