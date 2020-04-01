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


        stage("Notify") {
            agent { label "sl7" }
            steps {
                setGitHubBuildStatus("pending", "Build and tests are starting...")
                echo "Branch: ${env.JOB_BASE_NAME}"
            }
        }

        stage("UNIX environment") {

            agent { label "sl7" }

            stages {

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
                            python -m pip install numpy &&
                            python -m pip install matplotlib &&
                            python -m pip install tox==3.14.5 &&
                            python -m pip install pylint==2.4.4 &&
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

                stage("PyPI Release Testing"){
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

                stage("Set up") {
                    steps {
                        checkout scm
                        bat """
                            set CONDA="C:\\Programming\\miniconda3\\condabin\\conda.bat"
                            CALL %CONDA% config --append channels free
                            CALL %CONDA% create --name py python=3.6.0 -y
                            CALL %CONDA% activate py
                            python -m pip install --upgrade --user pip
                            python -m pip install numpy
                            python -m pip install matplotlib
                            python -m pip install tox==3.14.5
                            python -m pip install pylint==2.4.4
                        """
                    }
                }

                stage("Test VS2017") {
                    steps {
                        bat """
                            CALL C:\\Programming\\VS2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat x86_amd64
                            set CONDA="C:\\Programming\\miniconda3\\condabin\\conda.bat"
                            CALL %CONDA% activate py
                            python -m tox
                        """
                    }
                }

                stage("Test VS2019") {
                    steps {
                        bat """
                            CALL C:\\Programming\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat
                            set CONDA="C:\\Programming\\miniconda3\\condabin\\conda.bat"
                            CALL %CONDA% activate py
                            python -m tox
                        """
                    }
                }

                stage("PyPI Release Testing VS2019"){
                    when { tag "*" }
                    steps {
                        bat """
                            CALL C:\\Programming\\VS2017\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat x86_amd64
                            set CONDA="C:\\Programming\\miniconda3\\condabin\\conda.bat"
                            rmdir /s /q .tox
                            CALL %CONDA% activate py
                            set /p EUPHONIC_VERSION= < python euphonic/get_version.py
                            python -m tox -c release_tox.ini
                        """
                    }
                }

                stage("PyPI Release Testing VS2017"){
                    when { tag "*" }
                    steps {
                        bat """
                            CALL C:\\Programming\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat
                            set CONDA="C:\\Programming\\miniconda3\\condabin\\conda.bat"
                            rmdir /s /q .tox
                            CALL %CONDA% activate py
                            set /p EUPHONIC_VERSION= < python euphonic/get_version.py
                            python -m tox -c release_tox.ini
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            node("sl7"){
                junit 'tests_and_analysis/test/reports/junit_report*.xml'
            }
            node("PACE Windows (Private)"){
                junit 'tests_and_analysis/test/reports/junit_report*.xml'
            }
        }

        success {
            node("sl7"){
                setGitHubBuildStatus("success", "Linux: Successful")
            }
            node("PACE Windows (Private)"){
                setGitHubBuildStatus("success", "Windows: Successful")
            }
        }

        unsuccessful {
            node("sl7"){
                setGitHubBuildStatus("failure", "Linux: Unsuccessful")
            }
            node("PACE Windows (Private)"){
                setGitHubBuildStatus("success", "Windows: Unsuccessful")
            }
        }

        cleanup {
            node("sl7"){
                deleteDir()
            }
            node("PACE Windows (Private)"){
                deleteDir()
            }
        }
    }

}
