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

        stage("Checkout") {
            steps {
                setGitHubBuildStatus("pending", "Build and tests are starting...")
                echo "Branch: ${env.JOB_BASE_NAME}"
                checkout(
                    [
                        $class: 'GitSCM', branches: [[name: '${env.JOB_BASE_NAME}']],
                        doGenerateSubmoduleConfigurations: false,
                        extensions: [
                            [$class: 'RelativeTargetDirectory', relativeTargetDir: 'unix'],
                            [$class: 'CleanBeforeCheckout'], [$class: 'WipeWorkspace']
                        ],
                        submoduleCfg: [],
                        userRemoteConfigs: [[url: 'https://github.com/pace-neutrons/Euphonic']]
                    ]
                )
                checkout(
                    [
                        $class: 'GitSCM', branches: [[name: '${env.JOB_BASE_NAME}']],
                        doGenerateSubmoduleConfigurations: false,
                        extensions: [
                            [$class: 'RelativeTargetDirectory', relativeTargetDir: 'windows'],
                            [$class: 'CleanBeforeCheckout'], [$class: 'WipeWorkspace']
                        ],
                        submoduleCfg: [],
                        userRemoteConfigs: [[url: 'https://github.com/pace-neutrons/Euphonic']]
                    ]
                )
            }
        }

        stage("Set up: UNIX environment") {
            agent { label "sl7" }
            steps {
                sh """
                    pushd unix &&
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
                    export CC=gcc &&
                    popd
                """
            }
        }

        stage("Test: UNIX environment") {
            agent { label "sl7" }
            steps {
                sh """
                    pushd unix &&
                    module load conda/3 &&
                    conda config --append channels free &&
                    conda activate py &&
                    python -m tox &&
                    popd
                """
            }
        }

        stage("PyPI Release Testing: UNIX environment"){
            agent { label "sl7" }
            when { tag "*" }
            steps {
                sh """
                    pushd unix &&
                    rm -rf .tox &&
                    module load conda/3 &&
                    conda config --append channels free &&
                    conda activate py &&
                    export EUPHONIC_VERSION="\$(python euphonic/get_version.py)" &&
                    python -m tox -c release_tox.ini &&
                    popd
                """
            }
        }

        stage("Set up: Windows environment") {
            agent { label "PACE Windows (Private)" }
            steps {
                bat """
                    echo "Setting up windows env"
                """
            }
        }

        stage("Test: Windows environment") {
            agent { label "PACE Windows (Private)" }
            steps {
                bat """
                    echo "Testing on windows env"
                """
            }
        }

        stage("PyPI Release Testing: Windows environment"){
            agent { label "PACE Windows (Private)" }
            when { tag "*" }
            steps {
                bat """
                    echo "Release Testing on windows env"
                """
            }
        }

        stage("Static Code Analysis") {
            agent { label "sl7" }
            steps {
                sh """
                    module load conda/3 &&
                    conda config --append channels free &&
                    conda activate py &&
                    python tests_and_analysis/static_code_analysis/run_analysis.py
                """
                def pylint_issues = scanForIssues tool: pyLint(pattern: "tests_and_analysis/static_code_analysis/reports/pylint_output.txt")
                publishIssues issues: [pylint_issues]
            }
        }
    }

    post {
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
