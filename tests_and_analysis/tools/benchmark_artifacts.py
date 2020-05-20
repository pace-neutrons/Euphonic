import argparse
import requests
from typing import Dict, Tuple, Union, List
import os
import json


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
             "You can create a token on the Jenkins instance by clicking on"
             "your username in the top right hand corner and going to "
             "configure. This token requires admin access to the "
             "PACE-neutrons project, which can be requested "
             "from ANVIL@stfc.ac.uk"
    )
    # The location to copy artifacts to
    parser.add_argument(
        "-c", "--copy-to-location", action="store", dest="copy_to_location",
        type=str, default=(r'\\isis.cclrc.ac.uk\Shares\PACE_Project_Tool_Source'
                           r'\euphonic_performance_benchmarking'),
        help="The location to which you wish to copy your artifacts to"
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


def copy_artifacts(artifacts, jenkins_build_url, copy_to_location, timestamp,
                   user_id, token):
    for artifact in artifacts:
        if "performance_benchmarks.json" == artifact["fileName"]:
            performance_benchmarks_json = requests.get(
                jenkins_build_url + "artifact/" + artifact["relativePath"],
                auth=(user_id, token)
            ).json()
            # Write the json to file
            directory = os.path.join(copy_to_location, str(timestamp))
            os.mkdir(directory)
            json.dump(
                performance_benchmarks_json,
                open(
                    os.path.join(directory, "performance_benchmarks.json"),
                    "w+"
                )
            )


if __name__ == "__main__":
    args_parsed = get_parser().parse_args()
    user_id = args_parsed.user_id
    token = args_parsed.token
    performance_benchmarking_response = requests.get(
        "https://anvil.softeng-support.ac.uk/jenkins/job/"
        "PACE-neutrons/job/Euphonic/job/Performance Benchmarking/api/json/",
        auth=(user_id, token)
    )
    if performance_benchmarking_response.status_code == 404:
        raise Exception(
            """
                404 status code, likely authentication has failed.
                Please ensure the user_id and token used has admin access to 
                the Anvil Jenkins PACE-neutrons project. You can request this 
                from ANVIL@stfc.ac.uk
            """
        )
    lower_bound, upper_bound = coerce_range(
        performance_benchmarking_response.json(), args_parsed.range
    )
    copy_to_location = (r'\\isis.cclrc.ac.uk\Shares\PACE_Project_Tool_Source'
                        r'\euphonic_performance_benchmarking')
    for build in performance_benchmarking_response.json()["builds"]:
        if lower_bound <= build["number"] <= upper_bound:
            url = build["url"]
            build_response = requests.get(
                url + "/api/json",
                auth=(user_id, token)
            )
            if build_response.json()["result"] == "SUCCESS":
                copy_artifacts(
                    build_response.json()["artifacts"],
                    url,
                    copy_to_location,
                    build_response.json()["timestamp"],
                    user_id,
                    token
                )


