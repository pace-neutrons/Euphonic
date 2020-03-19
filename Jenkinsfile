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

        stage("Set up environment") {
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            module load conda/3 &&
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
                            conda activate py &&
                            python -m tox
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
            githubNotify status: "SUCCES", description: "Build was successful"
        }

        unsuccessful {
            githubNotify status: "FAILURE", description: "Build failed"
        }

        cleanup {
            deleteDir()
        }
    }

}
