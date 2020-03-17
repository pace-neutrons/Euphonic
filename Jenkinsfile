#!groovy

pipeline {

    // "sl7 && PACE Windows (Private)" for when windows is ready 
    agent { 
        label "sl7" 
    }

    triggers {
        pollSCM('H/2 * * * *')
    }

    parameters {
        string(
            name: 'PYTHON3_VERSION', defaultValue: '3.6.0', 
            description: 'The version of python 3 to build and test with'
        )

        string(
            name: 'PYTHON2_VERSION', defaultValue: '2.7.11', 
            description: 'The version of python 2 to build and test with'
        )

        string(
            name: 'CONDA3_VERSION', defaultValue: '3', 
            description: 'The version of conda to set up the python 3 environment with'
        )

        string(
            name: 'CONDA2_VERSION', defaultValue: '2', 
            description: 'The version of conda to set up the python 2 environment with'
        )
    }
  
    stages {

        stage("Checkout") {
            steps {
                echo "Branch: ${env.BRANCH_NAME}"
                checkout scm
            }
        }

        stage("Build") {
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            ./build_tools/jenkins_env_setup_linux.sh ${params.PYTHON3_VERSION} ${params.CONDA3_VERSION} &&
                            ./build_tools/jenkins_env_setup_linux.sh ${params.PYTHON2_VERSION} ${params.CONDA2_VERSION}
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
                            ./build_tools/activate_python_and_run_tests.sh ${params.PYTHON3_VERSION} ${params.CONDA3_VERSION} &&
                            ./build_tools/activate_python_and_run_tests.sh ${params.PYTHON2_VERSION} ${params.CONDA2_VERSION}
                        """
                    }
                }
            }
        }

    }
}
