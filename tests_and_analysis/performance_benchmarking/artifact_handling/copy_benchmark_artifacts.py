import argparse
import requests
from typing import Dict, Tuple, Union, List
import os
import json

jenkins_api_help_string = (
    "You can create a token on the Jenkins instance by clicking on "
    "your username in the top right hand corner and going to "
    "configure.\nThis token requires admin access to the "
    "PACE-neutrons project, which can be requested "
    "from ANVIL@stfc.ac.uk "
)


def get_parser():
    parser = argparse.ArgumentParser()
    # Log in details
    parser.add_argument(
        "-u", "--user-id", action="store", required=True,
        dest="user_id", help="Your Anvil Jenkins user id. Found by clicking "
                             "on your username in the top right hand and it "
                             "should be displayed near the centre of the page."
    )
    parser.add_argument(
        "-t", "--token", action="store", required=True, dest="token",
        help="An Anvil Jenkins api token generated with your account. "
             "{}".format(jenkins_api_help_string)
    )
    # The location to copy artifacts to
    parser.add_argument(
        "-c", "--copy-to-location", action="store", dest="copy_to_location",
        type=str, required=True,
        help="The location to which you wish to copy your artifacts to."
    )
    # The url of the performance benchmarking job
    parser.add_argument(
        "-j", "--jenkins-job-url", action="store", dest="jenkins_job_url",
        type=str,
        default="https://anvil.softeng-support.ac.uk/jenkins/job/"
                "PACE-neutrons/job/Euphonic/job/Performance Benchmarking"
                "/api/json/",
        help="The url of the jenkins job running the performance benchmarking "
             "that contains the artifacts to copy"
    )
    # Range of builds you wish to recover artifacts from
    parser.add_argument(
        "-r", "--range", action="store", nargs=2, dest="range", type=int,
        help="The range of Jenkins builds you wish to recover artifacts from."
    )
    return parser


def coerce(value: int, lower_bound: int, upper_bound: int) -> int:
    """
    Coerce the value into between the lower bound and upper bound (inclusive).

    Parameters
    ----------
    value : int
        The value to coerce
    lower_bound : int
        The inclusive lower bound to coerce the value into
    upper_bound : int
        The inclusive upper bound to coerce the value into

    Returns
    -------
    int
        The value coerced into being between the lower and upper bound.
    """
    return max(lower_bound, min(value, upper_bound))


def coerce_range(performance_benchmarking_response_json: Dict,
                 range_parsed: Union[None, List[int]]) -> Tuple[int, int]:
    """
    Take details of the builds and find the first and last build numbers.
    Coerce the range parsed by the user into between those first and last build
    numbers.

    Parameters
    ----------
    performance_benchmarking_response_json : Dict
        A json response from Jenkins with build details including firstBuild
        and lastBuild
    range_parsed : Union[None, List[int]]
        If no range was parsed this will be None, if a range has been parsed
        this will be a list of two integers.

    Returns
    -------
    Tuple[int, int]
        The first value is the range start point and the second is the range
        end point.
    """
    # The first and last build numbers
    largest_possible_range = (
        performance_benchmarking_response_json["firstBuild"]["number"] - 1,
        performance_benchmarking_response_json["lastBuild"]["number"]
    )
    if range_parsed:
        # Force both build numbers to be within the boundaries
        lower_build_number = coerce(
            range_parsed[0],
            largest_possible_range[0], largest_possible_range[1]
        )
        upper_build_number = coerce(
            range_parsed[1],
            largest_possible_range[0], largest_possible_range[1]
        )
        return lower_build_number, upper_build_number
    else:
        # When there is no range parsed copy all the builds
        return largest_possible_range


def write_to_file(directory: str, filename: str,
                  performance_benchmarks_json: str):
    try:
        os.mkdir(directory)
        filepath = os.path.join(directory, filename)
        json.dump(performance_benchmarks_json, open(filepath, "w+"))
    except FileExistsError:
        pass


def copy_benchmark_json(artifacts: List[Dict[str, str]], jenkins_build_url: str,
                        copy_to_location: str, timestamp: int,
                        user_id: str, token: str):
    """
    Copy a performance_benchmarks.json artifact from a Jenkins build to the
    given location in a directory with a name given by the timestamp.

    Parameters
    ----------
    artifacts : List[Dict[int, Dict[str, str]]]
        A list of artifacts from a Jenkins build including the
        performance_benchmarks.json artifact. Each artifact has the
        fields fileName and relativePath fields.
    jenkins_build_url : str
        The url of the Jenkins build the artifact is from and from which the
        relativePath of the artifact follows.
    copy_to_location : str
        A directory to which to copy the artifacts to.
    timestamp : int
        The timestamp from the Jenkins build (gives the directory name
        with which to write the json file to)
    user_id : str
        The Jenkins user id to use when pulling the artifact from Jenkins.
    token : str
        A Jenkins api token to user when pulling the artifact from Jenkins.
    """
    # Search for the performance_benchmarks.json artifact
    for artifact in artifacts:
        if "performance_benchmarks.json" == artifact["fileName"]:
            # Get the json to write to file
            performance_benchmarks_json = requests.get(
                jenkins_build_url + "artifact/" + artifact["relativePath"],
                auth=(user_id, token)
            ).json()
            # Locate where to copy the file to
            directory = os.path.join(copy_to_location, str(timestamp))
            filename = "performance_benchmarks.json"
            write_to_file(directory, filename, performance_benchmarks_json)


if __name__ == "__main__":
    # retrieve args
    args_parsed = get_parser().parse_args()
    user_id = args_parsed.user_id
    token = args_parsed.token
    copy_to_location = args_parsed.copy_to_location
    # Get builds details
    job_response = requests.get(
        "https://anvil.softeng-support.ac.uk/jenkins/job/"
        "PACE-neutrons/job/Euphonic/job/Performance Benchmarking/api/json/",
        auth=(user_id, token)
    )
    # If we get a failure response error the script
    if 100 <= job_response.status_code >= 300:
        raise Exception(
            "{} status code. Error when contacting Jenkins api.\n"
            "If 404, it is likely authentication has failed.\n"
            "{} ".format(
                job_response.status_code, jenkins_api_help_string
            )
        )
    # Coerce the parsed range into the range of jobs available
    lower_bound, upper_bound = coerce_range(
        job_response.json(), args_parsed.range
    )
    copied_artifact_builds = []
    # Only get artifacts from builds within the given range
    for build in job_response.json()["builds"]:
        if lower_bound <= build["number"] <= upper_bound:
            # Get details from the build
            url = build["url"]
            build_response = requests.get(
                url + "/api/json", auth=(user_id, token)
            )
            # Copy artifact if build was successful
            if build_response.json()["result"] == "SUCCESS":
                copy_benchmark_json(
                    build_response.json()["artifacts"], url,
                    copy_to_location, build_response.json()["timestamp"],
                    user_id, token
                )
                copied_artifact_builds.append(build["number"])
    copied_artifact_builds.sort()
    print("Success copied artifacts for builds: {}".format(
        str(copied_artifact_builds)
    ))
