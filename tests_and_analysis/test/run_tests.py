from argparse import ArgumentParser, Namespace
import os
import sys
import time
from uuid import uuid4

import pytest


def main():
    test_dir, reports_dir = _get_test_and_reports_dir()

    args = _get_parsed_args(test_dir)

    pytest_options: list[str] = _build_pytest_options(
        reports_dir=reports_dir,
        do_report_tests=args.report,
        tests=args.test_file,
        markers=args.markers_to_run,
        parallel=args.parallel,
    )

    test_exit_code: int = run_tests(
        pytest_options, args.cov, reports_dir, test_dir)

    # Exit with a failure code if there are any errors or failures
    sys.exit(test_exit_code)


def _get_test_and_reports_dir() -> tuple[str, str]:
    """
    Get the directory that holds the tests and the directory to write
    reports to. If the directory to write reports to isn't present, it
    is created.

    Returns
    -------
    str
        The directory the tests are situated in
    str
        The directory to write reports to
    """
    test_dir: str = os.path.dirname(os.path.abspath(__file__))
    reports_dir: str = os.path.join(test_dir, 'reports')
    if not os.path.exists(reports_dir):
        os.mkdir(reports_dir)
    return test_dir, reports_dir

def _get_parsed_args(test_dir: str) -> Namespace:
    """
    Get the arguments parsed to this script.

    Parameters
    ----------

    test_dir: test directory (for default test file search)
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--cov', action='store_true',
        help='If present report coverage in a coverage*.xml file in reports')
    parser.add_argument(
        '--report', action='store_true',
        help='If present report test results to junit_report*.xml files')
    parser.add_argument('-t', '--test-file', dest='test_file', action='store',
                        help='The test file to run', default=test_dir)
    parser.add_argument(
        '-m', action='store', dest='markers_to_run',
        help=('Limit the test runs to only the specified markers e.g.'
              'e.g. "unit" or "unit or integration"'), default='')
    parser.add_argument(
        '--parallel', action='store_true',
        help='Use pytest-xdist for parallel testing',
    )
    return parser.parse_args()


def _build_pytest_options(
        *,
        reports_dir: str,
        do_report_tests: bool,
        tests: str,
        markers: str,
        parallel: bool,
) -> list[str]:
    """
    Build the options for pytest to use.

    Parameters
    ----------
    reports_dir : str
        The directory to write reports to (for the junit xml reports)
    do_report_tests : bool
        Whether to write the test reports to junit xml or not.
    tests : str
        The tests to run e.g. script_tests or test_bands_data.py
    markers : str|None
        The markers for pytest tests to run e.g. "unit" or "unit or
        integration"

    Returns
    -------
    list[str]
        A list of options to run pytest with.
    """
    options: list[str] = [tests]
    # Add reporting of test results
    if do_report_tests:
        # We may have multiple reports, so get a unique filename
        filename_prefix = 'junit_report'
        filenum = int(time.time())
        junit_xml_filepath = os.path.join(
            reports_dir, f'{filename_prefix}_{filenum}.xml')
        options.append(f'--junitxml={junit_xml_filepath}')
    # Only run the specified markers
    options.append(f'-m={markers}')

    if parallel:
        options.append('--numprocesses=auto')
    return options


def run_tests(pytest_options: list[str], do_report_coverage: bool,
              reports_dir: str, test_dir: str) -> int:
    """
    Run the tests and record coverage if selected.

    Parameters
    ----------
    pytest_options : list[str]
        The options to pass to pytest
    do_report_coverage : bool
        If true report coverage to coverage*.xml
    reports_dir : str
        The directory to report coverage to
    test_dir: str
        The directory where the tests are stored

    Returns
    -------
    A test exit code. 0 is success, 1 to 5 are all errors
     (see pytest docs for further details).
    """
    # Set import-mode to ensure the installed version is tested rather
    # than the local version
    pytest_options = ['--import-mode=append', *pytest_options]

    # Start recording coverage if requested
    if do_report_coverage:
        coveragerc_filepath: str = os.path.join(test_dir, '.coveragerc')

        pytest_options = [
            '--cov',
            f'--cov-report=xml:{reports_dir}/coverage_{uuid4()}.xml',
            f'--cov-config={coveragerc_filepath}',
            *pytest_options,
        ]

    # Run tests and get the resulting exit code
    # 0 is success, 1-5 are different forms of failure (see pytest docs
    # for details)
    return pytest.main(pytest_options)


if __name__ == '__main__':
    main()
