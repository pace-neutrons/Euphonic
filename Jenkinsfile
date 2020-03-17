#!groovy

versions = [[python_version: "2.7.11", conda_version: "2"], [python_version: "3.6.0", conda_version: "2"]]



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
           for (version in versions) {
                steps {
                    script {
                        if (isUnix()) {
                            sh """
                                module load conda/${version['conda_version']} &&
                                module load gcc &&
                                conda create --name py${version['python_version']} python=${version['python_version']} -y &&
                                conda activate py${version['python_version']} &&
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
        }

        stage("Test") {
            for (version in versions) {
                steps {
                    script {
                        if (isUnix()) {
                            sh """
                                module load conda/${version['conda_version']} &&
                                conda activate py${version['python_version']} &&
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
}
