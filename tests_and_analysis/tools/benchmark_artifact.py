import jenkins
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--username", action="store", required=True,
        dest="username", help="Your Anvil Jenkins username"
    )
    parser.add_argument(
        "-t", "--token", action="store", required=True, dest="token",
        help="An Anvil Jenkins api token associated with your account"
    )
    return parser


if __name__ == "__main__":
    args_parsed = get_parser().parse_args()
    server = jenkins.Jenkins(
        "https://anvil.softeng-support.ac.uk/jenkins",
        username=args_parsed.username, password=args_parsed.token
    )
    user = server.get_whoami()
    version = server.get_version()
    print('Hello %s from Jenkins %s' % (user['fullName'], version))