#!groovy

void setGitHubBuildStatus(String status, message) {
    script {
        withCredentials([string(credentialsId: 'GitHub_API_Token',
                variable: 'api_token')]) {
            if (isUnix()) {
                sh """
                    curl -H "Authorization: token ${api_token}" \
                    --request POST \
                    --data '{"state": "${status}", \
                        "description": "${message}", \
                        "target_url": "$BUILD_URL", \
                        "context": "$JOB_BASE_NAME"}' \
                    https://api.github.com/repos/pace-neutrons/Euphonic/statuses/${env.GIT_COMMIT}
                """
            }
        }
    }
}

pipeline {

    agent none

    triggers {
        GenericTrigger(
             genericVariables: [
                [key: 'ref', value: '$.ref']
             ],

             causeString: 'Triggered on $ref',

             token: 'GitHub_API_Token',

             printContributedVariables: true,
             printPostContent: true,

             silentResponse: false,

             regexpFilterText: '$ref',
             regexpFilterExpression: 'refs/head/' + env.JOB_BASE_NAME
        )
        pollSCM('')
    }

    stages {

        stage("UNIX environment") {

            agent { label "sl7" }

            stages {

                stage("Checkout: UNIX environment") {
                    steps {
                        setGitHubBuildStatus("pending", "Build and tests are starting...")
                        echo "Branch: ${env.JOB_BASE_NAME}"
                        checkout scm
                    }
                }

                stage("Set up: UNIX environment") {
                    steps {
                        sh """
                            module load conda/3 &&
                            conda config --append channels free &&
                            module load gcc &&
                            conda create --name py python=3.6.0 -y &&
                            conda activate py &&
                            python -m pip install --upgrade --user pip &&
                            python -m pip install numpy &&
                            python -m pip install matplotlib &&
                            python -m pip install tox==3.14.5 &&
                            python -m pip install pylint==2.4.4 &&
                            export CC=gcc
                        """
                    }
                }

                stage("Test: UNIX environment") {
                    steps {
                        sh """
                            module load conda/3 &&
                            conda config --append channels free &&
                            conda activate py &&
                            python -m tox
                        """
                    }
                }

                stage("PyPI Release Testing: UNIX environment"){
                    when { tag "*" }
                    steps {
                        sh """
                            rm -rf .tox &&
                            module load conda/3 &&
                            conda config --append channels free &&
                            conda activate py &&
                            export EUPHONIC_VERSION="\$(python euphonic/get_version.py)" &&
                            python -m tox -c release_tox.ini
                        """
                    }
                }

                stage("Static Code Analysis") {
                    steps {
                        sh """
                            module load conda/3 &&
                            conda config --append channels free &&
                            conda activate py &&
                            python tests_and_analysis/static_code_analysis/run_analysis.py
                        """
                        script {
                            def pylint_issues = scanForIssues tool: pyLint(pattern: "tests_and_analysis/static_code_analysis/reports/pylint_output.txt")
                            publishIssues issues: [pylint_issues]
                        }
                    }
                }
            }

        }

        stage("Windows environment") {

            agent { label "PACE Windows (Private)" }

            stages {

                stage("Checkout: Windows environment") {
                    steps {
                        echo "Branch: ${env.JOB_BASE_NAME}"
                        checkout scm
                    }
                }

                stage("Set up: Windows environment") {
                    steps {
                        bat """
                            echo "Setting up windows env"
                        """
                    }
                }

                stage("Test: Windows environment") {
                    steps {
                        bat """
                            echo "Testing on windows env"
                        """
                    }
                }

                stage("PyPI Release Testing: Windows environment"){
                    when { tag "*" }
                    steps {
                        bat """
                            echo "Release Testing on windows env"
                        """
                    }
                }
            }
        }
    }

    post {
        node("sl7"){
            always {
                junit 'tests_and_analysis/test/reports/junit_report*.xml'
            }

            success {
                setGitHubBuildStatus("success", "Build and tests were successful")
            }

            unsuccessful {
                setGitHubBuildStatus("failure", "Build or tests have failed")
            }

            cleanup {
                deleteDir()
            }
        }

        node("PACE Windows (Private)"){
            always {
                junit 'tests_and_analysis/test/reports/junit_report*.xml'
            }

            success {
                setGitHubBuildStatus("success", "Build and tests were successful")
            }

            unsuccessful {
                setGitHubBuildStatus("failure", "Build or tests have failed")
            }

            cleanup {
                deleteDir()
            }
        }
    }

}
