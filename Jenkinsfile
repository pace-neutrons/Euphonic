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
            name: 'python_version', defaultValue: '3.6.0', 
            description: 'The version of python to build and test with'
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
                            module load conda/3 &&
                            module load gcc &&
                            conda create --name py python=${params.python_version} -y &&
                            conda activate py &&
                            export CC=gcc &&
                            python -m pip install --upgrade --user pip &&
                            python -m pip install --user numpy &&
                            python -m pip install --user .[matplotlib] &&
                            python -m pip install --user mock
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
                            conda activate py &&
                            pushd test &&
                            python -m unittest discover -v . &&
                            popd
                        """
                    }
                }
            }
        }

    }
}
