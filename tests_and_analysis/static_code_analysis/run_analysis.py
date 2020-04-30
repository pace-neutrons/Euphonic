# Attributes to: https://stackoverflow.com/questions/2028268/invoking-pylint-programmatically

from pylint import lint
from pylint.reporters.text import ParseableTextReporter
import os
import sys


class LintOutput(object):
    """An output consumer for the linter."""

    def __init__(self):
        self.content = []

    def write(self, string):
        """
        Write to the output.

        Parameters
        ----------
        string : str
            The string to write to the output
        """
        self.content.append(string)

    def read(self):
        """Read all the ouput from the linting process."""
        return self.content

    def write_to_file(self, filename):
        """
        Write the linting output to file.

        Parameters
        ----------
        filename : str
            The name of the file to write the output to.
        """
        with open(filename, "w+") as f:
            f.writelines(self.content)


if __name__ == "__main__":

    # The directory to change back to at the end
    original_cwd = os.getcwd()

    # Change directory to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Options to control pylint process
    pylint_options = ["-r", "n", "--rcfile=.pylintrc"]

    # The files and directories to lint
    files_and_dirs = list(map(lambda file_or_dir: "../../{}".format(file_or_dir),
                              ["euphonic", "scripts", "release.py", "setup.py"]))

    # Object to write pylint output to
    pylint_output = LintOutput()

    # Run the linting
    run = lint.Run(files_and_dirs+pylint_options, reporter=ParseableTextReporter(pylint_output), do_exit=False)

    # Write the lint output to file
    os.makedirs("reports", exist_ok=True)
    pylint_output.write_to_file("reports/pylint_output.txt")

    # Move back to original directory
    os.chdir(original_cwd)

    # If we have a score lower than the threshold fail the linting
    threshold = 10

    score = round(run.linter.stats['global_note'], 2)

    if score < threshold:
        print("Score ({}) is less than threshold ({}). Analysis will report failure. "
              "Correct issues or adjust threshold.".format(score, threshold))
        sys.exit(run.linter.msg_status)
