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

    // "sl7 && PACE Windows (Private)" for when windows is ready
    agent {
        label "sl7"
    }

    triggers {
        pollSCM('H/2 * * * *')
    }

    stages {

        stage("Checkout") {
            steps {
                setGitHubBuildStatus("pending", "Build and tests are starting...")
                echo "Branch: ${env.BRANCH_NAME}"
                checkout scm
            }
        }

        stage("Set up environment") {
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            module load conda/3 &&
                            conda config --append channels free &&
                            module load gcc &&
                            conda create --name py python=3.6.0 -y &&
                            conda activate py &&
                            python -m pip install --upgrade --user pip &&
                            python -m pip install tox &&
                            export CC=gcc
                        """
                    }
                }
            }
        }

        stage("Test") {
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            module load conda/3 &&
                            conda config --append channels free &&
                            conda activate py &&
                            python -m tox
                        """
                    }
                }
            }
        }

        stage("PyPI Release Testing") {
            when { tag "*" }
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            rm -rf .tox &&
                            module load conda/3 &&
                            conda config --append channels free &&
                            conda activate py &&
                            export EUPHONIC_VERSION="$(python euphonic/get_version.py)" &&
                            python -m tox -c release_tox.ini
                        """
                    }
                }
            }
        }

    }

    post {
        always {
            junit 'test/reports/**/*.xml'
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
