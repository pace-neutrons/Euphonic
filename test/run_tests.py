import sys
import os
import unittest
import xmlrunner

if __name__ == "__main__":

    # Discover tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite = unittest.TestLoader().discover(test_dir, pattern="test_*.py")

    # Set output directory and run tests
    xml_dir = os.path.join(test_dir, "reports")
    ret_vals = xmlrunner.XMLTestRunner(output=xml_dir).run(test_suite)

    sys.exit(bool(ret_vals.errors or ret_vals.failures))
