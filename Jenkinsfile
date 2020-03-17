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
                            export PYTHON_VERSION=${params.python_version}
                            pushd build_scripts
                            ./conda_jenkins_build.sh $PYTHON_VERSION
                            popd
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
