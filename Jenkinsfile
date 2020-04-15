#!groovy

def setGitHubBuildStatus(String status, String message, String context) {
    script {
        withCredentials([string(credentialsId: 'Euphonic_GitHub_API_Token',
                variable: 'api_token')]) {
            if (isUnix()) {
                sh """
                    curl -H "Authorization: token ${api_token}" \
                    --request POST \
                    --data '{ \
                        "state": "${status}", \
                        "description": "${message} on ${context}", \
                        "target_url": "$BUILD_URL", \
                        "context": "jenkins/${context}" \
                    }' \
                    https://api.github.com/repos/pace-neutrons/Euphonic/statuses/${env.GIT_COMMIT}
                """
            } else {
                powershell """
                    [Net.ServicePointManager]::SecurityProtocol = "tls12, tls11, tls"
                    \$payload = @{
                      "state" = "${status}";
                      "description" = "${message} on ${context}";
                      "target_url" = "$BUILD_URL";
                      "context" = "jenkins/${context}"}
                    Invoke-RestMethod -URI "https://api.github.com/repos/pace-neutrons/Euphonic/statuses/${env.GIT_COMMIT}" \
                      -Headers @{Authorization = "token ${api_token}"} \
                      -Method 'POST' \
                      -Body (\$payload|ConvertTo-JSON) \
                      -ContentType "application/json"
                  """
            }
        }
    }
}

def getGithubCommitAuthorEmail(){
    script {
        withCredentials([string(credentialsId: 'Euphonic_GitHub_API_Token',
                variable: 'api_token')]) {
            if (isUnix()) {
                email = sh script: """
                    payload=\$(curl --silent -H "Authorization: token ${api_token}" --request GET \
                        https://api.github.com/repos/pace-neutrons/Euphonic/commits/${env.GIT_COMMIT} > /dev/null) &&
                    email=\$payload | jq -r ".commit.author.email" &&
                    echo \$email
                """, returnStdout: true
            } else {
                email = powershell script: """
                    [Net.ServicePointManager]::SecurityProtocol = "tls12, tls11, tls"
                    \$commit_payload = Invoke-RestMethod \
                        -URI "https://api.github.com/repos/pace-neutrons/Euphonic/commits/${env.GIT_COMMIT}" \
                        -Headers @{Authorization = "token ${api_token}"} \
                        -Method 'GET'
                    echo \$commit_payload
                    \$email = \$commit_payload.commit.author.email
                    echo \$email
                  """, returnStdout: true
            }
        }
    }
    return email.trim()
}

pipeline {

    agent none

    triggers {
        GenericTrigger(
             genericVariables: [
                [key: 'ref', value: '$.ref']
             ],

             causeString: 'Triggered on $ref',

             token: 'Euphonic_GitHub_API_Token',

             printContributedVariables: true,
             printPostContent: true,

             silentResponse: false,

             regexpFilterText: '$ref',
             regexpFilterExpression: 'refs/head/' + env.JOB_BASE_NAME
        )
        pollSCM('')
    }

    stages {

        stage("Parallel environments") {

            parallel {

                stage("UNIX environment") {

                    agent { label "sl7" }

                    stages {

                        stage("Notify") {
                            steps {
                                script {
                                    def email = getGithubCommitAuthorEmail()
                                }
                                echo "$email"
                                setGitHubBuildStatus("pending", "Starting", "Linux")
                                echo "Branch: ${env.JOB_BASE_NAME}"
                            }
                        }

                        stage("Set up") {
                            steps {
                                checkout scm
                                sh """
                                    module load conda/3 &&
                                    conda config --append channels free &&
                                    module load gcc &&
                                    conda create --name py python=3.6.0 -y &&
                                    conda activate py &&
                                    python -m pip install --upgrade --user pip &&
                                    python -m pip install -r tests_and_analysis/jenkins_requirements.txt &&
                                    export CC=gcc
                                """
                            }
                        }

                        stage("Test") {
                            steps {
                                sh """
                                    module load conda/3 &&
                                    conda config --append channels free &&
                                    conda activate py &&
                                    python -m tox
                                """
                            }
                        }

                        stage("PyPI Release Testing") {
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

                    post {

                        always {
                            junit 'tests_and_analysis/test/reports/junit_report*.xml'
                        
                            publishCoverage adapters: [coberturaAdapter('tests_and_analysis/test/reports/coverage.xml')]
                        }

                        success {
                            setGitHubBuildStatus("success", "Successful", "Linux")
                        }

                        unsuccessful {
                            setGitHubBuildStatus("failure", "Unsuccessful", "Linux")
                        }

                        cleanup {
                            deleteDir()
                        }

                    }
                }

                stage("Windows environment") {

                    agent { label "PACE Windows (Private)" }

                    stages {

                        stage("Notify") {
                            steps {
                                script {
                                    def email = getGithubCommitAuthorEmail()
                                }
                                echo "$email"
                                setGitHubBuildStatus("pending", "Starting", "Windows")
                                echo "Branch: ${env.JOB_BASE_NAME}"
                            }
                        }

                        stage("Set up") {
                            steps {
                                checkout scm
                                bat """
                                    CALL conda create --name py python=3.6.0 -y
                                    CALL conda activate py
                                    python -m pip install --upgrade --user pip
                                    python -m pip install numpy
                                    python -m pip install matplotlib
                                    python -m pip install tox==3.14.5
                                    python -m pip install pylint==2.4.4
                                """
                            }
                        }

                        stage("Test VS2019") {
                            steps {
                                bat """
                                    CALL "%VS2019_VCVARSALL%" x86_amd64
                                    CALL conda activate py
                                    python -m tox
                                """
                            }
                        }

                        stage("PyPI Release Testing VS2019") {
                            when { tag "*" }
                            steps {
                                bat """
                                    CALL "%VS2019_VCVARSALL%" x86_amd64
                                    rmdir /s /q .tox
                                    CALL conda activate py
                                    set /p EUPHONIC_VERSION= < python euphonic/get_version.py
                                    python -m tox -c release_tox.ini
                                """
                            }
                        }
                    }

                    post {

                        always {
                            junit 'tests_and_analysis/test/reports/junit_report*.xml'
                        }

                        success {
                            setGitHubBuildStatus("success", "Successful", "Windows")
                        }

                        unsuccessful {
                            setGitHubBuildStatus("failure", "Unsuccessful", "Windows")
                        }

                        cleanup {
                            deleteDir()
                        }

                    }
                }
            }
        }
    }
}
