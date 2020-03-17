#!groovy

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
                echo "Branch: ${env.BRANCH_NAME}"
                checkout scm
            }
        }

        stage("Build") {
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            module load conda/3
                            module load gcc


                            conda create --name py python=3.6.0 -y
                            conda activate py

                            export CC=gcc

                            python -m pip install --upgrade --user pip
                            python -m pip install --user numpy
                            python -m pip install --user .[matplotlib]
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
                            pushd Euphonic/test
                            python -m unittest discover -v .
                            popd
                        """
                    }
                }
            }
        }

    }
}
