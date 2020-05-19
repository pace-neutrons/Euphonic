import argparse
from pprint import pprint
import requests
from typing import Dict, Tuple, Union, List


def get_parser():
    parser = argparse.ArgumentParser()
    # Log in details
    parser.add_argument(
        "-u", "--username", action="store", required=True,
        dest="username", help="Your Anvil Jenkins username"
    )
    parser.add_argument(
        "-t", "--token", action="store", required=True, dest="token",
        help="An Anvil Jenkins api token generated with your account. "
             "This token requires admin access to the PACE-neutrons project, "
             "which can be requested from ANVIL@stfc.ac.uk"
    )
    # Range of builds you wish to recover artifacts from
    parser.add_argument(
        "-r", "--range", action="store", nargs=2, dest="range", type=int,
        help="The range of Jenkins builds you wish to recover artifacts from."
    )
    return parser


def coerce(value, lower_bound, upper_bound):
    return max(lower_bound, min(value, upper_bound))


def coerce_range(performance_benchmarking_response_json: Dict,
                 range_parsed: Union[None, List[int]]) -> Tuple[int, int]:
    largest_possible_range = (
        performance_benchmarking_response_json["firstBuild"]["number"] - 1,
        performance_benchmarking_response_json["lastBuild"]["number"]
    )
    if range_parsed:
        lower_index = coerce(
            range_parsed[0],
            largest_possible_range[0], largest_possible_range[1]
        )
        upper_index = coerce(
            range_parsed[1],
            largest_possible_range[0], largest_possible_range[1]
        )
        return lower_index, upper_index
    else:
        return largest_possible_range


if __name__ == "__main__":
    args_parsed = get_parser().parse_args()
    performance_benchmarking_response = requests.get(
        "https://anvil.softeng-support.ac.uk/jenkins/job/"
        "PACE-neutrons/job/Euphonic/job/Performance Benchmarking/api/json/",
        auth=(args_parsed.username, args_parsed.token)
    )
    if performance_benchmarking_response.status_code == 404:
        raise Exception(
            """
                404 status code, likely authentication has failed.
                Please ensure the username and token used has admin access to 
                the Anvil Jenkins PACE-neutrons project. You can request this 
                from ANVIL@stfc.ac.uk
            """
        )
    coerced_range = coerce_range(
        performance_benchmarking_response.json(), args_parsed.range
    )
    for build in range(*coerced_range):
        url = performance_benchmarking_response.json()["builds"][build]["url"]
        build_response = requests.get(
            url + "/api/json",
            auth=(args_parsed.username, args_parsed.token)
        )
        if build_response.json()["result"] == "SUCCESS":
            for artifact in build_response.json()["artifacts"]:
                if "performance_benchmarks.json" == artifact["fileName"]:
                    performance_benchmarks_json = requests.get(
                        url + "artifact/" + artifact["relativePath"],
                        auth=(args_parsed.username, args_parsed.token)
                    )
                    pprint(performance_benchmarks_json.json()["speedups"])

