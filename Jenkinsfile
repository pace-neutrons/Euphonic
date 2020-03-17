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
                        sh '''
                            ./build/conda_jenkins_build.sh
                        '''
                    } else {
                        bat '''
                            echo "TODO: set up windows and macOS nodes" 
                        '''
                    }
                }
            }
        }

        stage("Test") {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            module load python/\$PYTHON_VERSION &&
                            pushd Euphonic/test
                            python -m unittest discover -v .
                            popd
                        '''
                    } else {
                        bat '''
                            echo "TODO: set up windows and macOS nodes" 
                        '''
                    }
                }
            }
        }

    }
}
