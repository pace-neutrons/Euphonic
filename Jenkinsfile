#!groovy

def post_github_status(String state, String message) {
  // Non-PR builds will not set PR_STATUSES_URL - in which case we do not
  // want to post any statuses to Git
  if (env.PR_STATUSES_URL) {
    script {
      withCredentials([string(credentialsId: 'GitHub_API_Token',
          variable: 'api_token')]) {
        if (isUnix()) {
          sh """
            curl -H "Authorization: token ${api_token}" \
              --request POST \
              --data '{"state": "${state}", \
                "description": "${message}", \
                "target_url": "$BUILD_URL", \
                "context": "$JOB_BASE_NAME"}' \
              $PR_STATUSES_URL > /dev/null
            """
        }
        else {
          powershell """
            [Net.ServicePointManager]::SecurityProtocol = "tls12, tls11, tls"
            \$payload = @{
              "state" = "${state}";
              "description" = "${message}";
              "target_url" = "$BUILD_URL";
              "context" = "$JOB_BASE_NAME"}
            Invoke-RestMethod -URI "$PR_STATUSES_URL" \
              -Headers @{Authorization = "token ${api_token}"} \
              -Method 'POST' \
              -Body (\$payload|ConvertTo-JSON) \
              -ContentType "application/json"
          """
        }
      }
    }
  }
}

def setBuildStatus(String state, String message) {
    script {
        withCredentials([string(credentialsId: 'GitHub_API_Token',
          variable: 'api_token')]) {

        }
    }
    if (isUnix()) {
        sh """
            curl -XPOST -H "Authorization: token ${api_token}" https://api.github.com/repos/:pace-neutrons/:Euphonic/statuses/$(git rev-parse HEAD) -d "{
                \"state\": \"success\",
                \"target_url\": \"${BUILD_URL}\",
                \"description\": \"The build has succeeded!\"
            }"
        """
    }
}

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
                setBuildStatus("pending", "The build is pending")
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
            setBuildStatus("success", "The build and tests succeeded")
        }

        unsuccessful {
            setBuildStatus("failure", "The build or tests failed")
        }

        cleanup {
            deleteDir()
        }
    }

}
