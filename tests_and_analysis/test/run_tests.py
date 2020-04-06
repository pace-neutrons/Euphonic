import sys
import os
import pytest
import coverage
import argparse

if __name__ == "__main__":

    # Check whether the recording coverage has been requested
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", action="store_true")
    parser.add_argument("--report", action="store_true")
    args_parsed = parser.parse_args()
    do_record_coverage = args_parsed.cov
    do_report_results = args_parsed.report

    # Set output directory to the reports directory under this file's directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    xml_dir = os.path.join(test_dir, "reports")
    if not os.path.exists(xml_dir):
        os.mkdir(xml_dir)

    # Start recording coverage if requested
    cov = None
    if do_record_coverage:
        coveragerc_filepath = os.path.join(test_dir, ".coveragerc")
        cov = coverage.Coverage(config_file=coveragerc_filepath)
        cov.start()

    # We may have multiple reports, so do not overwrite them
    filename_prefix = "junit_report"
    filenum = 0
    for filename in os.listdir(xml_dir):
        if filename_prefix in filename:
            filenum += 1
    xml_filepath = os.path.join(test_dir, "reports", "{}{}.xml".format(filename_prefix, filenum))

    # Run tests and get the resulting exit code
    # 0 is success, 1-5 are different forms of failure (see pytest docs for details)
    if do_report_results:
        test_exit_code = pytest.main([test_dir, "--junitxml={}".format(xml_filepath)])
    else:
        test_exit_code = pytest.main([test_dir])

    # Report coverage if requested
    if do_record_coverage and cov is not None:
        cov.stop()
        coverage_xml_filepath = os.path.join(xml_dir, "coverage.xml")
        cov.xml_report(outfile=coverage_xml_filepath)

    # Exit with a failure code if there are any errors or failures
    sys.exit(test_exit_code)
